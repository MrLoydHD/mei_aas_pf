"""
DistilBERT-based model for DGA detection.
Uses pre-trained DistilBERT fine-tuned for binary classification (PyTorch).
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


class DistilBERTDGADetector:
    """
    DistilBERT-based DGA detector using pre-trained language model (PyTorch).

    This model fine-tunes a pre-trained DistilBERT model for binary
    domain classification. DistilBERT is a smaller, faster version
    of BERT that retains 97% of its language understanding capabilities.
    """

    def __init__(
        self,
        max_length: int = 128,
        learning_rate: float = 2e-5,
        model_name: str = 'distilbert-base-uncased'
    ):
        """
        Initialize the DistilBERT DGA detector.

        Args:
            max_length: Maximum token sequence length
            learning_rate: Learning rate for fine-tuning
            model_name: Pre-trained model to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFace Transformers or PyTorch not installed. "
                "Install with: pip install transformers torch"
            )

        self.max_length = max_length
        self.learning_rate = learning_rate
        self.model_name = model_name

        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.model: Optional[DistilBertForSequenceClassification] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        self.history: Dict[str, List[float]] = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def _load_pretrained(self) -> None:
        """Load pre-trained tokenizer and model."""
        print(f"Loading pre-trained {self.model_name}...")
        print(f"Using device: {self.device}")

        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.to(self.device)

    def tokenize_domains(self, domains: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize domain names for DistilBERT.

        Args:
            domains: List of domain names

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call _load_pretrained() first.")

        # Add spaces between characters for better tokenization
        spaced_domains = [' '.join(list(d.lower())) for d in domains]

        encoded = self.tokenizer(
            spaced_domains,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def prepare_data(
        self,
        dga_domains: List[str],
        legit_domains: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Prepare tokenized data and labels.

        Args:
            dga_domains: List of DGA domain names
            legit_domains: List of legitimate domain names

        Returns:
            Tuple of (tokenized_data, labels)
        """
        if self.tokenizer is None:
            self._load_pretrained()

        all_domains = dga_domains + legit_domains
        tokenized = self.tokenize_domains(all_domains)
        y = np.array([1] * len(dga_domains) + [0] * len(legit_domains))

        return tokenized, y

    def train(
        self,
        X: Dict[str, torch.Tensor],
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 3,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Fine-tune DistilBERT for DGA detection.

        Args:
            X: Tokenized input (dict with input_ids and attention_mask)
            y: Labels
            test_size: Proportion for testing
            epochs: Training epochs (typically 2-4 for fine-tuning)
            batch_size: Training batch size

        Returns:
            Dictionary of training metrics
        """
        print("Splitting data...")

        # Split indices
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y
        )

        # Create datasets
        train_dataset = TensorDataset(
            X['input_ids'][train_idx],
            X['attention_mask'][train_idx],
            torch.tensor(y[train_idx], dtype=torch.long)
        )
        test_dataset = TensorDataset(
            X['input_ids'][test_idx],
            X['attention_mask'][test_idx],
            torch.tensor(y[test_idx], dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Load model if not already loaded
        if self.model is None:
            self._load_pretrained()

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        print(f"\nFine-tuning on {len(train_idx)} samples...")

        best_val_loss = float('inf')
        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            self.history['loss'].append(avg_train_loss)
            self.history['accuracy'].append(train_acc)

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    preds = torch.argmax(outputs.logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(test_loader)
            val_acc = val_correct / val_total
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"loss: {avg_train_loss:.4f} - acc: {train_acc:.4f} - "
                  f"val_loss: {avg_val_loss:.4f} - val_acc: {val_acc:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.is_trained = True

        # Evaluate
        print("\nEvaluating model...")
        self.metrics = self.evaluate(
            {'input_ids': X['input_ids'][test_idx], 'attention_mask': X['attention_mask'][test_idx]},
            y[test_idx]
        )

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall: {self.metrics['recall']:.4f}")
        print(f"  F1-Score: {self.metrics['f1']:.4f}")
        print(f"  ROC-AUC: {self.metrics['roc_auc']:.4f}")

        return self.metrics

    def evaluate(self, X: Dict[str, torch.Tensor], y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X: Tokenized input
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        dataset = TensorDataset(X['input_ids'], X['attention_mask'])
        loader = DataLoader(dataset, batch_size=32)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        probs = np.vstack(all_probs)
        y_prob = probs[:, 1]
        y_pred = np.argmax(probs, axis=1)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True)
        }

    def predict(self, domain: str) -> Dict[str, Any]:
        """
        Predict if a single domain is DGA-generated.

        Args:
            domain: Domain name to classify

        Returns:
            Dictionary with prediction and confidence
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        tokenized = self.tokenize_domains([domain])

        with torch.no_grad():
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        is_dga = probs[1] > probs[0]
        confidence = float(probs[1] if is_dga else probs[0])

        return {
            'domain': domain,
            'is_dga': bool(is_dga),
            'confidence': confidence,
            'dga_probability': float(probs[1]),
            'legit_probability': float(probs[0])
        }

    def predict_batch(self, domains: List[str]) -> List[Dict[str, Any]]:
        """
        Predict for multiple domains.

        Args:
            domains: List of domain names

        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self.model.eval()
        tokenized = self.tokenize_domains(domains)

        dataset = TensorDataset(tokenized['input_ids'], tokenized['attention_mask'])
        loader = DataLoader(dataset, batch_size=32)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        probs = np.vstack(all_probs)

        results = []
        for i, domain in enumerate(domains):
            is_dga = probs[i, 1] > probs[i, 0]
            confidence = float(probs[i, 1] if is_dga else probs[i, 0])
            results.append({
                'domain': domain,
                'is_dga': bool(is_dga),
                'confidence': confidence,
                'dga_probability': float(probs[i, 1]),
                'legit_probability': float(probs[i, 0])
            })

        return results

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history metrics."""
        return self.history

    def save(self, path: str) -> None:
        """
        Save the fine-tuned model.

        Args:
            path: Directory path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        os.makedirs(path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(os.path.join(path, 'model'))
        self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))

        # Save metadata
        import json
        metadata = {
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name,
            'metrics': self.metrics,
            'history': self.history
        }

        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a fine-tuned model.

        Args:
            path: Directory path containing the saved model
        """
        import json

        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            os.path.join(path, 'tokenizer')
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            os.path.join(path, 'model')
        )
        self.model.to(self.device)

        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.max_length = metadata['max_length']
        self.learning_rate = metadata['learning_rate']
        self.model_name = metadata['model_name']
        self.metrics = metadata['metrics']

        if 'history' in metadata:
            self.history = metadata['history']

        self.is_trained = True
        print(f"Model loaded from {path}")
