"""
Transformer-based deep learning model for DGA detection.
Uses character-level embeddings with self-attention for classification.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
import keras
from keras import layers, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import pad_sequences

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


class TransformerBlock(layers.Layer):
    """Transformer encoder block with multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Multi-head self-attention with residual connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for sequence position information."""

    def __init__(self, max_length: int, embed_dim: int):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        # Compute positional encoding matrix
        position = np.arange(max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        pe = np.zeros((max_length, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pos_encoding = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]


class TransformerDGADetector:
    """
    Transformer-based DGA detector using character-level embeddings.

    This model uses self-attention mechanisms to capture long-range
    dependencies in domain name character sequences, often achieving
    higher accuracy than LSTM-based models.
    """

    # Character vocabulary (same as LSTM for consistency)
    CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789-_.'
    CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for padding
    IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
    VOCAB_SIZE = len(CHARS) + 1  # +1 for padding

    def __init__(
        self,
        max_length: int = 63,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001
    ):
        """
        Initialize the Transformer DGA detector.

        Args:
            max_length: Maximum domain name length
            embedding_dim: Character embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of Transformer encoder layers
            ff_dim: Feed-forward network dimension
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model: Optional[Model] = None
        self.is_trained = False
        self.metrics: Dict[str, Any] = {}
        self.history: Optional[Any] = None

    def _build_model(self) -> Model:
        """Build the Transformer model architecture."""
        inputs = layers.Input(shape=(self.max_length,), name='input')

        # Character embedding
        x = layers.Embedding(
            input_dim=self.VOCAB_SIZE,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)

        # Positional encoding
        x = PositionalEncoding(self.max_length, self.embedding_dim)(x)

        # Transformer encoder blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(x)

        # Global average pooling to aggregate sequence information
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)

        # Classification head
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(self.dropout_rate / 2, name='dropout_2')(x)

        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs, name='transformer_dga_detector')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        return model

    def encode_domain(self, domain: str) -> np.ndarray:
        """
        Encode a domain name to integer sequence.

        Args:
            domain: Domain name string

        Returns:
            Numpy array of character indices
        """
        domain_lower = domain.lower()
        encoded = [self.CHAR_TO_IDX.get(c, 0) for c in domain_lower]
        return np.array(encoded)

    def encode_domains(self, domains: List[str]) -> np.ndarray:
        """
        Encode multiple domains with padding.

        Args:
            domains: List of domain names

        Returns:
            Padded numpy array of shape (n_samples, max_length)
        """
        encoded = [self.encode_domain(d) for d in domains]
        return pad_sequences(encoded, maxlen=self.max_length, padding='post', truncating='post')

    def prepare_data(
        self,
        dga_domains: List[str],
        legit_domains: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare encoded data and labels.

        Args:
            dga_domains: List of DGA domain names
            legit_domains: List of legitimate domain names

        Returns:
            Tuple of (encoded_domains, labels)
        """
        all_domains = dga_domains + legit_domains
        X = self.encode_domains(all_domains)
        y = np.array([1] * len(dga_domains) + [0] * len(legit_domains))

        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 256
    ) -> Dict[str, Any]:
        """
        Train the Transformer model.

        Args:
            X: Encoded domain sequences
            y: Labels
            test_size: Proportion for testing
            epochs: Maximum training epochs
            batch_size: Training batch size

        Returns:
            Dictionary of training metrics
        """
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Build model
        print("Building Transformer model...")
        self.model = self._build_model()
        self.model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        print(f"\nTraining on {len(X_train)} samples...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
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

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X: Encoded sequences
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_prob = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)

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

        encoded = self.encode_domains([domain])
        prob = self.model.predict(encoded, verbose=0)[0][0]
        is_dga = prob > 0.5

        return {
            'domain': domain,
            'is_dga': bool(is_dga),
            'confidence': float(prob if is_dga else 1 - prob),
            'dga_probability': float(prob),
            'legit_probability': float(1 - prob)
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

        encoded = self.encode_domains(domains)
        probs = self.model.predict(encoded, verbose=0).flatten()

        results = []
        for i, domain in enumerate(domains):
            prob = probs[i]
            is_dga = prob > 0.5
            results.append({
                'domain': domain,
                'is_dga': bool(is_dga),
                'confidence': float(prob if is_dga else 1 - prob),
                'dga_probability': float(prob),
                'legit_probability': float(1 - prob)
            })

        return results

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history metrics."""
        if self.history is None:
            return {}
        return self.history.history

    def save(self, path: str) -> None:
        """
        Save the trained model.

        Args:
            path: Directory path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        os.makedirs(path, exist_ok=True)

        # Save Keras model
        self.model.save(os.path.join(path, 'model.keras'))

        # Save metadata
        import json
        metadata = {
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'metrics': self.metrics,
            'history': self.get_training_history()
        }

        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Directory path containing the saved model
        """
        import json

        # Load Keras model with custom objects
        self.model = keras.models.load_model(
            os.path.join(path, 'model.keras'),
            custom_objects={
                'TransformerBlock': TransformerBlock,
                'PositionalEncoding': PositionalEncoding
            }
        )

        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.max_length = metadata['max_length']
        self.embedding_dim = metadata['embedding_dim']
        self.num_heads = metadata['num_heads']
        self.num_layers = metadata['num_layers']
        self.ff_dim = metadata['ff_dim']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.metrics = metadata['metrics']

        # Load training history if available
        if 'history' in metadata:
            self.metrics['history'] = metadata['history']

        self.is_trained = True
        print(f"Model loaded from {path}")
