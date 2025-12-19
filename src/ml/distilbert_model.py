"""
DistilBERT-based model for DGA detection.
Uses pre-trained DistilBERT fine-tuned for binary classification.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
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

# HuggingFace Transformers imports
try:
    from transformers import (
        DistilBertTokenizer,
        TFDistilBertForSequenceClassification,
        DistilBertConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class DistilBERTDGADetector:
    """
    DistilBERT-based DGA detector using pre-trained language model.

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
                "HuggingFace Transformers not installed. "
                "Install with: pip install transformers"
            )

        self.max_length = max_length
        self.learning_rate = learning_rate
        self.model_name = model_name

        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.model: Optional[TFDistilBertForSequenceClassification] = None
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        self.history: Optional[Any] = None

    def _load_pretrained(self) -> None:
        """Load pre-trained tokenizer and model."""
        print(f"Loading pre-trained {self.model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Configure for binary classification
        config = DistilBertConfig.from_pretrained(
            self.model_name,
            num_labels=2,
            output_hidden_states=False,
            output_attentions=False
        )

        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )

    def tokenize_domains(self, domains: List[str]) -> Dict[str, tf.Tensor]:
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
        # This helps BERT-based models handle domain-like text
        spaced_domains = [' '.join(list(d.lower())) for d in domains]

        encoded = self.tokenizer(
            spaced_domains,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def prepare_data(
        self,
        dga_domains: List[str],
        legit_domains: List[str]
    ) -> Tuple[Dict[str, tf.Tensor], np.ndarray]:
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
        X: Dict[str, tf.Tensor],
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

        # Split data
        X_train = {
            'input_ids': tf.gather(X['input_ids'], train_idx),
            'attention_mask': tf.gather(X['attention_mask'], train_idx)
        }
        X_test = {
            'input_ids': tf.gather(X['input_ids'], test_idx),
            'attention_mask': tf.gather(X['attention_mask'], test_idx)
        }
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Load model if not already loaded
        if self.model is None:
            self._load_pretrained()

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )

        print(f"\nFine-tuning on {len(y_train)} samples...")

        # Create TF datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
            y_train
        )).shuffle(len(y_train)).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            {'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']},
            y_test
        )).batch(batch_size)

        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True

        # Evaluate
        print("\nEvaluating model...")
        self.metrics = self.evaluate(X_test, y_test)

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall: {self.metrics['recall']:.4f}")
        print(f"  F1-Score: {self.metrics['f1']:.4f}")
        print(f"  ROC-AUC: {self.metrics['roc_auc']:.4f}")

        return self.metrics

    def evaluate(self, X: Dict[str, tf.Tensor], y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X: Tokenized input
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        outputs = self.model.predict(X, verbose=0)
        logits = outputs.logits

        # Convert to probabilities
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        y_prob = probs[:, 1]  # Probability of DGA class
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

        tokenized = self.tokenize_domains([domain])
        outputs = self.model.predict(tokenized, verbose=0)
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

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

        tokenized = self.tokenize_domains(domains)
        outputs = self.model.predict(tokenized, verbose=0)
        probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()

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
        if self.history is None:
            return {}
        return self.history.history

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
            'history': self.get_training_history()
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
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            os.path.join(path, 'model')
        )

        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.max_length = metadata['max_length']
        self.learning_rate = metadata['learning_rate']
        self.model_name = metadata['model_name']
        self.metrics = metadata['metrics']

        # Load training history if available
        if 'history' in metadata:
            self.metrics['history'] = metadata['history']

        self.is_trained = True
        print(f"Model loaded from {path}")
