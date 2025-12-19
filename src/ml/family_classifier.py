"""
DGA Family Classification Model.
Classifies DGA domains into specific malware family categories.
Used as a second-stage classifier after binary DGA detection.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src.ml.features import DomainFeatureExtractor


# DGA Family metadata for threat intelligence
DGA_FAMILY_INFO = {
    'conficker': {
        'description': 'Conficker worm DGA',
        'threat_level': 'high',
        'first_seen': '2008',
        'type': 'worm'
    },
    'cryptolocker': {
        'description': 'CryptoLocker ransomware DGA',
        'threat_level': 'critical',
        'first_seen': '2013',
        'type': 'ransomware'
    },
    'emotet': {
        'description': 'Emotet banking trojan DGA',
        'threat_level': 'critical',
        'first_seen': '2014',
        'type': 'banking_trojan'
    },
    'necurs': {
        'description': 'Necurs botnet DGA',
        'threat_level': 'high',
        'first_seen': '2012',
        'type': 'botnet'
    },
    'gozi': {
        'description': 'Gozi/Ursnif banking trojan DGA',
        'threat_level': 'high',
        'first_seen': '2007',
        'type': 'banking_trojan'
    },
    'ramnit': {
        'description': 'Ramnit banking trojan DGA',
        'threat_level': 'high',
        'first_seen': '2010',
        'type': 'banking_trojan'
    },
    'matsnu': {
        'description': 'Matsnu backdoor DGA',
        'threat_level': 'medium',
        'first_seen': '2014',
        'type': 'backdoor'
    },
    'suppobox': {
        'description': 'Suppobox trojan DGA',
        'threat_level': 'medium',
        'first_seen': '2015',
        'type': 'trojan'
    },
    'ranbyus': {
        'description': 'Ranbyus banking trojan DGA',
        'threat_level': 'high',
        'first_seen': '2013',
        'type': 'banking_trojan'
    },
    'simda': {
        'description': 'Simda botnet DGA',
        'threat_level': 'medium',
        'first_seen': '2012',
        'type': 'botnet'
    },
    'pushdo': {
        'description': 'Pushdo/Cutwail spam botnet DGA',
        'threat_level': 'medium',
        'first_seen': '2007',
        'type': 'botnet'
    },
    'dircrypt': {
        'description': 'DirCrypt ransomware DGA',
        'threat_level': 'high',
        'first_seen': '2014',
        'type': 'ransomware'
    },
    'tinba': {
        'description': 'Tinba/Tiny Banker trojan DGA',
        'threat_level': 'high',
        'first_seen': '2012',
        'type': 'banking_trojan'
    },
    'vawtrak': {
        'description': 'Vawtrak/Neverquest banking trojan DGA',
        'threat_level': 'high',
        'first_seen': '2013',
        'type': 'banking_trojan'
    },
    'symmi': {
        'description': 'Symmi trojan DGA',
        'threat_level': 'medium',
        'first_seen': '2014',
        'type': 'trojan'
    },
    'nymaim': {
        'description': 'Nymaim ransomware/downloader DGA',
        'threat_level': 'high',
        'first_seen': '2013',
        'type': 'ransomware'
    },
    'kraken': {
        'description': 'Kraken botnet DGA',
        'threat_level': 'medium',
        'first_seen': '2008',
        'type': 'botnet'
    },
    'qadars': {
        'description': 'Qadars banking trojan DGA',
        'threat_level': 'high',
        'first_seen': '2013',
        'type': 'banking_trojan'
    },
    'corebot': {
        'description': 'CoreBot banking trojan DGA',
        'threat_level': 'high',
        'first_seen': '2015',
        'type': 'banking_trojan'
    },
    'rovnix': {
        'description': 'Rovnix bootkit DGA',
        'threat_level': 'high',
        'first_seen': '2011',
        'type': 'bootkit'
    },
    'murofet': {
        'description': 'Murofet trojan DGA',
        'threat_level': 'medium',
        'first_seen': '2010',
        'type': 'trojan'
    },
    'fobber': {
        'description': 'Fobber banking trojan DGA',
        'threat_level': 'high',
        'first_seen': '2015',
        'type': 'banking_trojan'
    },
    'ramdo': {
        'description': 'Ramdo click-fraud malware DGA',
        'threat_level': 'low',
        'first_seen': '2013',
        'type': 'adware'
    },
    'pykspa': {
        'description': 'Pykspa worm DGA',
        'threat_level': 'medium',
        'first_seen': '2012',
        'type': 'worm'
    },
    'padcrypt': {
        'description': 'PadCrypt ransomware DGA',
        'threat_level': 'high',
        'first_seen': '2016',
        'type': 'ransomware'
    },
}


class DGAFamilyClassifier:
    """
    Random Forest classifier for DGA family classification.
    Classifies DGA domains into specific malware families.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        word_dict_path: Optional[str] = None,
        legit_domains_path: Optional[str] = None,
        use_tfidf: bool = True
    ):
        """
        Initialize the family classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            word_dict_path: Path to word dictionary for feature extraction
            legit_domains_path: Path to legitimate domains for TF-IDF
            use_tfidf: Whether to use TF-IDF features
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_extractor = None
        self.feature_names = None
        self.families = None
        self.metrics = None

        if word_dict_path or legit_domains_path:
            self.feature_extractor = DomainFeatureExtractor(
                word_dict_path=word_dict_path,
                legit_domains_path=legit_domains_path,
                use_tfidf=use_tfidf
            )

    def prepare_data(
        self,
        domains: List[str],
        families: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data by extracting features.

        Args:
            domains: List of DGA domain names
            families: List of corresponding family labels

        Returns:
            Tuple of (features array, encoded labels array)
        """
        print(f"Extracting features from {len(domains)} domains...")
        features_df = self.feature_extractor.extract_features_batch(domains)
        self.feature_names = self.feature_extractor.get_feature_names()

        # Encode family labels
        y = self.label_encoder.fit_transform(families)
        self.families = list(self.label_encoder.classes_)

        print(f"Found {len(self.families)} unique families")
        print(f"Extracted {len(self.feature_names)} features")

        return features_df.values, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train the family classifier.

        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Dictionary of evaluation metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=self.families,
                output_dict=True
            ),
            'num_families': len(self.families),
            'families': self.families
        }

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {self.metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {self.metrics['recall_macro']:.4f}")
        print(f"  F1-Score (macro): {self.metrics['f1_macro']:.4f}")

        return self.metrics

    def predict(self, domain: str) -> Dict:
        """
        Predict the DGA family for a single domain.

        Args:
            domain: Domain name to classify

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Extract features
        features = self.feature_extractor.extract_features(domain)
        X = np.array([list(features.values())])

        # Get prediction and probabilities
        pred_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        family = self.families[pred_idx]
        confidence = float(proba[pred_idx])

        # Get top 3 predictions
        top_indices = np.argsort(proba)[::-1][:3]
        alternatives = [
            {
                'family': self.families[idx],
                'confidence': float(proba[idx])
            }
            for idx in top_indices[1:]  # Exclude top prediction
        ]

        # Get family info
        family_info = DGA_FAMILY_INFO.get(family, {
            'description': f'{family} DGA',
            'threat_level': 'unknown',
            'first_seen': 'unknown',
            'type': 'unknown'
        })

        return {
            'family': family,
            'confidence': confidence,
            'description': family_info['description'],
            'threat_level': family_info['threat_level'],
            'first_seen': family_info['first_seen'],
            'malware_type': family_info['type'],
            'alternatives': alternatives
        }

    def predict_batch(self, domains: List[str]) -> List[Dict]:
        """
        Predict families for multiple domains.

        Args:
            domains: List of domain names

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(d) for d in domains]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None or self.feature_names is None:
            return {}

        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def save(self, path: str) -> None:
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_extractor': self.feature_extractor,
            'feature_names': self.feature_names,
            'families': self.families,
            'metrics': self.metrics
        }
        joblib.dump(model_data, path)
        print(f"Family classifier saved to {path}")

    def load(self, path: str) -> None:
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_extractor = model_data['feature_extractor']
        self.feature_names = model_data['feature_names']
        self.families = model_data['families']
        self.metrics = model_data.get('metrics')
        print(f"Family classifier loaded from {path}")
        print(f"Families: {len(self.families)}")


class LSTMFamilyClassifier:
    """
    LSTM-based classifier for DGA family classification.
    Uses character-level embeddings for sequence modeling.
    """

    def __init__(
        self,
        max_length: int = 63,
        embedding_dim: int = 64,
        lstm_units: int = 128,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the LSTM family classifier.

        Args:
            max_length: Maximum domain length
            embedding_dim: Embedding dimension
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate
        """
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.char_to_idx = None
        self.label_encoder = LabelEncoder()
        self.families = None
        self.metrics = None
        self.history = None

    def _build_char_mapping(self, domains: List[str]) -> None:
        """Build character to index mapping."""
        chars = set()
        for domain in domains:
            chars.update(domain.lower())
        chars = sorted(chars)
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = len(self.char_to_idx)

    def _encode_domain(self, domain: str) -> np.ndarray:
        """Encode domain as sequence of character indices."""
        domain = domain.lower()[:self.max_length]
        encoded = [
            self.char_to_idx.get(c, self.char_to_idx['<UNK>'])
            for c in domain
        ]
        # Pad sequence
        if len(encoded) < self.max_length:
            encoded = encoded + [0] * (self.max_length - len(encoded))
        return np.array(encoded)

    def prepare_data(
        self,
        domains: List[str],
        families: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data.

        Args:
            domains: List of domain names
            families: List of family labels

        Returns:
            Tuple of (encoded domains, encoded labels)
        """
        print(f"Preparing {len(domains)} domains for LSTM training...")

        # Build character mapping
        self._build_char_mapping(domains)

        # Encode domains
        X = np.array([self._encode_domain(d) for d in domains])

        # Encode labels
        y = self.label_encoder.fit_transform(families)
        self.families = list(self.label_encoder.classes_)

        print(f"Vocabulary size: {len(self.char_to_idx)}")
        print(f"Number of families: {len(self.families)}")

        return X, y

    def build_model(self, num_classes: int) -> None:
        """Build the LSTM model."""
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        vocab_size = len(self.char_to_idx)

        inputs = layers.Input(shape=(self.max_length,))

        # Embedding
        x = layers.Embedding(vocab_size, self.embedding_dim)(inputs)

        # CNN for local patterns
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)

        # Bidirectional LSTM
        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(self.lstm_units // 2))(x)

        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output - softmax for multi-class
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 30,
        batch_size: int = 256
    ) -> Dict:
        """
        Train the LSTM family classifier.

        Args:
            X: Encoded domain sequences
            y: Labels
            test_size: Test set fraction
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Evaluation metrics
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Build model
        num_classes = len(self.families)
        self.build_model(num_classes)

        print(f"Training LSTM on {len(X_train)} samples...")
        self.model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3)
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        # Evaluate
        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'num_families': len(self.families),
            'families': self.families,
            'history': self.history
        }

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  F1-Score (macro): {self.metrics['f1_macro']:.4f}")

        return self.metrics

    def predict(self, domain: str) -> Dict:
        """Predict family for a single domain."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        X = np.array([self._encode_domain(domain)])
        proba = self.model.predict(X, verbose=0)[0]

        pred_idx = np.argmax(proba)
        family = self.families[pred_idx]
        confidence = float(proba[pred_idx])

        # Get top 3
        top_indices = np.argsort(proba)[::-1][:3]
        alternatives = [
            {'family': self.families[idx], 'confidence': float(proba[idx])}
            for idx in top_indices[1:]
        ]

        family_info = DGA_FAMILY_INFO.get(family, {
            'description': f'{family} DGA',
            'threat_level': 'unknown',
            'first_seen': 'unknown',
            'type': 'unknown'
        })

        return {
            'family': family,
            'confidence': confidence,
            'description': family_info['description'],
            'threat_level': family_info['threat_level'],
            'first_seen': family_info['first_seen'],
            'malware_type': family_info['type'],
            'alternatives': alternatives
        }

    def save(self, path: str) -> None:
        """Save the model."""
        os.makedirs(path, exist_ok=True)

        # Save Keras model
        self.model.save(os.path.join(path, 'model.keras'))

        # Save metadata
        metadata = {
            'char_to_idx': self.char_to_idx,
            'families': self.families,
            'metrics': self.metrics,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f"LSTM family classifier saved to {path}")

    def load(self, path: str) -> None:
        """Load the model."""
        import tensorflow as tf

        # Load Keras model
        self.model = tf.keras.models.load_model(os.path.join(path, 'model.keras'))

        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.char_to_idx = metadata['char_to_idx']
        self.families = metadata['families']
        self.metrics = metadata.get('metrics')
        self.max_length = metadata['max_length']

        print(f"LSTM family classifier loaded from {path}")
        print(f"Families: {len(self.families)}")


class XGBoostFamilyClassifier:
    """
    XGBoost classifier for DGA family classification.
    Uses handcrafted features with gradient boosting.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        learning_rate: float = 0.1,
        word_dict_path: Optional[str] = None,
        legit_domains_path: Optional[str] = None,
        use_tfidf: bool = True
    ):
        from xgboost import XGBClassifier

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_extractor = None
        self.feature_names = None
        self.families = None
        self.metrics = None

        if word_dict_path or legit_domains_path:
            self.feature_extractor = DomainFeatureExtractor(
                word_dict_path=word_dict_path,
                legit_domains_path=legit_domains_path,
                use_tfidf=use_tfidf
            )

    def prepare_data(
        self,
        domains: List[str],
        families: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data by extracting features."""
        print(f"Extracting features from {len(domains)} domains...")
        features_df = self.feature_extractor.extract_features_batch(domains)
        self.feature_names = self.feature_extractor.get_feature_names()

        y = self.label_encoder.fit_transform(families)
        self.families = list(self.label_encoder.classes_)

        print(f"Found {len(self.families)} unique families")
        return features_df.values, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """Train the XGBoost family classifier."""
        from xgboost import XGBClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training XGBoost on {len(X_train)} samples...")

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            objective='multi:softprob'
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.families, output_dict=True
            ),
            'num_families': len(self.families),
            'families': self.families
        }

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  F1-Score (macro): {self.metrics['f1_macro']:.4f}")

        return self.metrics

    def predict(self, domain: str) -> Dict:
        """Predict the DGA family for a single domain."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        features = self.feature_extractor.extract_features(domain)
        X = np.array([list(features.values())])

        pred_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        family = self.families[pred_idx]
        confidence = float(proba[pred_idx])

        top_indices = np.argsort(proba)[::-1][:3]
        alternatives = [
            {'family': self.families[idx], 'confidence': float(proba[idx])}
            for idx in top_indices[1:]
        ]

        family_info = DGA_FAMILY_INFO.get(family, {
            'description': f'{family} DGA',
            'threat_level': 'unknown',
            'first_seen': 'unknown',
            'type': 'unknown'
        })

        return {
            'family': family,
            'confidence': confidence,
            'description': family_info['description'],
            'threat_level': family_info['threat_level'],
            'first_seen': family_info['first_seen'],
            'malware_type': family_info['type'],
            'alternatives': alternatives
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None or self.feature_names is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save(self, path: str) -> None:
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_extractor': self.feature_extractor,
            'feature_names': self.feature_names,
            'families': self.families,
            'metrics': self.metrics
        }
        joblib.dump(model_data, path)
        print(f"XGBoost family classifier saved to {path}")

    def load(self, path: str) -> None:
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_extractor = model_data['feature_extractor']
        self.feature_names = model_data['feature_names']
        self.families = model_data['families']
        self.metrics = model_data.get('metrics')
        print(f"XGBoost family classifier loaded from {path}")


class GradientBoostingFamilyClassifier:
    """
    Gradient Boosting classifier for DGA family classification.
    Uses sklearn's GradientBoostingClassifier with handcrafted features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        word_dict_path: Optional[str] = None,
        legit_domains_path: Optional[str] = None,
        use_tfidf: bool = True
    ):
        from sklearn.ensemble import GradientBoostingClassifier as GBC
        from sklearn.multiclass import OneVsRestClassifier

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_extractor = None
        self.feature_names = None
        self.families = None
        self.metrics = None

        if word_dict_path or legit_domains_path:
            self.feature_extractor = DomainFeatureExtractor(
                word_dict_path=word_dict_path,
                legit_domains_path=legit_domains_path,
                use_tfidf=use_tfidf
            )

    def prepare_data(
        self,
        domains: List[str],
        families: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data by extracting features."""
        print(f"Extracting features from {len(domains)} domains...")
        features_df = self.feature_extractor.extract_features_batch(domains)
        self.feature_names = self.feature_extractor.get_feature_names()

        y = self.label_encoder.fit_transform(families)
        self.families = list(self.label_encoder.classes_)

        print(f"Found {len(self.families)} unique families")
        return features_df.values, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """Train the Gradient Boosting family classifier."""
        from sklearn.ensemble import GradientBoostingClassifier as GBC

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training Gradient Boosting on {len(X_train)} samples...")

        self.model = GBC(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=random_state,
            n_iter_no_change=10,
            validation_fraction=0.1
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.families, output_dict=True
            ),
            'num_families': len(self.families),
            'families': self.families
        }

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  F1-Score (macro): {self.metrics['f1_macro']:.4f}")

        return self.metrics

    def predict(self, domain: str) -> Dict:
        """Predict the DGA family for a single domain."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        features = self.feature_extractor.extract_features(domain)
        X = np.array([list(features.values())])

        pred_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        family = self.families[pred_idx]
        confidence = float(proba[pred_idx])

        top_indices = np.argsort(proba)[::-1][:3]
        alternatives = [
            {'family': self.families[idx], 'confidence': float(proba[idx])}
            for idx in top_indices[1:]
        ]

        family_info = DGA_FAMILY_INFO.get(family, {
            'description': f'{family} DGA',
            'threat_level': 'unknown',
            'first_seen': 'unknown',
            'type': 'unknown'
        })

        return {
            'family': family,
            'confidence': confidence,
            'description': family_info['description'],
            'threat_level': family_info['threat_level'],
            'first_seen': family_info['first_seen'],
            'malware_type': family_info['type'],
            'alternatives': alternatives
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None or self.feature_names is None:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save(self, path: str) -> None:
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_extractor': self.feature_extractor,
            'feature_names': self.feature_names,
            'families': self.families,
            'metrics': self.metrics
        }
        joblib.dump(model_data, path)
        print(f"Gradient Boosting family classifier saved to {path}")

    def load(self, path: str) -> None:
        """Load the model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_extractor = model_data['feature_extractor']
        self.feature_names = model_data['feature_names']
        self.families = model_data['families']
        self.metrics = model_data.get('metrics')
        print(f"Gradient Boosting family classifier loaded from {path}")


class TransformerFamilyClassifier:
    """
    Transformer-based classifier for DGA family classification.
    Uses character-level embeddings with self-attention.
    """

    def __init__(
        self,
        max_length: int = 63,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout_rate: float = 0.1
    ):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.model = None
        self.char_to_idx = None
        self.label_encoder = LabelEncoder()
        self.families = None
        self.metrics = None
        self.history = None

    def _build_char_mapping(self, domains: List[str]) -> None:
        """Build character to index mapping."""
        chars = set()
        for domain in domains:
            chars.update(domain.lower())
        chars = sorted(chars)
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = len(self.char_to_idx)

    def _encode_domain(self, domain: str) -> np.ndarray:
        """Encode domain as sequence of character indices."""
        domain = domain.lower()[:self.max_length]
        encoded = [
            self.char_to_idx.get(c, self.char_to_idx['<UNK>'])
            for c in domain
        ]
        if len(encoded) < self.max_length:
            encoded = encoded + [0] * (self.max_length - len(encoded))
        return np.array(encoded)

    def prepare_data(
        self,
        domains: List[str],
        families: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data."""
        print(f"Preparing {len(domains)} domains for Transformer training...")

        self._build_char_mapping(domains)
        X = np.array([self._encode_domain(d) for d in domains])
        y = self.label_encoder.fit_transform(families)
        self.families = list(self.label_encoder.classes_)

        print(f"Vocabulary size: {len(self.char_to_idx)}")
        print(f"Number of families: {len(self.families)}")

        return X, y

    def build_model(self, num_classes: int) -> None:
        """Build the Transformer model."""
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        # Import custom layers from transformer_model
        from src.ml.transformer_model import TransformerBlock, PositionalEncoding

        vocab_size = len(self.char_to_idx)

        inputs = layers.Input(shape=(self.max_length,))

        # Embedding
        x = layers.Embedding(vocab_size, self.embedding_dim)(inputs)

        # Positional encoding
        x = PositionalEncoding(self.max_length, self.embedding_dim)(x)

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.embedding_dim, self.num_heads,
                self.ff_dim, self.dropout_rate
            )(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output - softmax for multi-class
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 30,
        batch_size: int = 256
    ) -> Dict:
        """Train the Transformer family classifier."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        num_classes = len(self.families)
        self.build_model(num_classes)

        print(f"Training Transformer on {len(X_train)} samples...")
        self.model.summary()

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'num_families': len(self.families),
            'families': self.families,
            'history': self.history
        }

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  F1-Score (macro): {self.metrics['f1_macro']:.4f}")

        return self.metrics

    def predict(self, domain: str) -> Dict:
        """Predict family for a single domain."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        X = np.array([self._encode_domain(domain)])
        proba = self.model.predict(X, verbose=0)[0]

        pred_idx = np.argmax(proba)
        family = self.families[pred_idx]
        confidence = float(proba[pred_idx])

        top_indices = np.argsort(proba)[::-1][:3]
        alternatives = [
            {'family': self.families[idx], 'confidence': float(proba[idx])}
            for idx in top_indices[1:]
        ]

        family_info = DGA_FAMILY_INFO.get(family, {
            'description': f'{family} DGA',
            'threat_level': 'unknown',
            'first_seen': 'unknown',
            'type': 'unknown'
        })

        return {
            'family': family,
            'confidence': confidence,
            'description': family_info['description'],
            'threat_level': family_info['threat_level'],
            'first_seen': family_info['first_seen'],
            'malware_type': family_info['type'],
            'alternatives': alternatives
        }

    def save(self, path: str) -> None:
        """Save the model."""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'model.keras'))

        metadata = {
            'char_to_idx': self.char_to_idx,
            'families': self.families,
            'metrics': self.metrics,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f"Transformer family classifier saved to {path}")

    def load(self, path: str) -> None:
        """Load the model."""
        import tensorflow as tf
        from src.ml.transformer_model import TransformerBlock, PositionalEncoding

        self.model = tf.keras.models.load_model(
            os.path.join(path, 'model.keras'),
            custom_objects={
                'TransformerBlock': TransformerBlock,
                'PositionalEncoding': PositionalEncoding
            }
        )

        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.char_to_idx = metadata['char_to_idx']
        self.families = metadata['families']
        self.metrics = metadata.get('metrics')
        self.max_length = metadata['max_length']

        print(f"Transformer family classifier loaded from {path}")


class DistilBERTFamilyClassifier:
    """
    DistilBERT-based classifier for DGA family classification.
    Uses pre-trained DistilBERT fine-tuned for multi-class classification.
    """

    def __init__(
        self,
        max_length: int = 128,
        learning_rate: float = 2e-5,
        model_name: str = 'distilbert-base-uncased'
    ):
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.families = None
        self.metrics = None
        self.history = None

    def _load_pretrained(self, num_classes: int) -> None:
        """Load pre-trained tokenizer and model."""
        from transformers import (
            DistilBertTokenizer,
            TFDistilBertForSequenceClassification,
            DistilBertConfig
        )

        print(f"Loading pre-trained {self.model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        config = DistilBertConfig.from_pretrained(
            self.model_name,
            num_labels=num_classes,
            output_hidden_states=False
        )

        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        )

    def tokenize_domains(self, domains: List[str]):
        """Tokenize domain names for DistilBERT."""
        import tensorflow as tf

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
        domains: List[str],
        families: List[str]
    ) -> Tuple[Dict, np.ndarray]:
        """Prepare tokenized data and labels."""
        from transformers import DistilBertTokenizer

        if self.tokenizer is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        y = self.label_encoder.fit_transform(families)
        self.families = list(self.label_encoder.classes_)

        print(f"Preparing {len(domains)} domains for DistilBERT training...")
        print(f"Number of families: {len(self.families)}")

        tokenized = self.tokenize_domains(domains)
        return tokenized, y

    def train(
        self,
        X: Dict,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 3,
        batch_size: int = 32
    ) -> Dict:
        """Fine-tune DistilBERT for family classification."""
        import tensorflow as tf

        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y
        )

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

        num_classes = len(self.families)
        if self.model is None:
            self._load_pretrained(num_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        print(f"\nFine-tuning DistilBERT on {len(y_train)} samples...")

        train_dataset = tf.data.Dataset.from_tensor_slices((
            {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
            y_train
        )).shuffle(len(y_train)).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            {'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']},
            y_test
        )).batch(batch_size)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        ]

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        self.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        # Evaluate
        outputs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(outputs.logits, axis=1)

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'num_families': len(self.families),
            'families': self.families,
            'history': self.history
        }

        print(f"\nResults:")
        print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  F1-Score (macro): {self.metrics['f1_macro']:.4f}")

        return self.metrics

    def predict(self, domain: str) -> Dict:
        """Predict family for a single domain."""
        import tensorflow as tf

        if self.model is None:
            raise ValueError("Model not trained or loaded")

        tokenized = self.tokenize_domains([domain])
        outputs = self.model.predict(tokenized, verbose=0)
        proba = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

        pred_idx = np.argmax(proba)
        family = self.families[pred_idx]
        confidence = float(proba[pred_idx])

        top_indices = np.argsort(proba)[::-1][:3]
        alternatives = [
            {'family': self.families[idx], 'confidence': float(proba[idx])}
            for idx in top_indices[1:]
        ]

        family_info = DGA_FAMILY_INFO.get(family, {
            'description': f'{family} DGA',
            'threat_level': 'unknown',
            'first_seen': 'unknown',
            'type': 'unknown'
        })

        return {
            'family': family,
            'confidence': confidence,
            'description': family_info['description'],
            'threat_level': family_info['threat_level'],
            'first_seen': family_info['first_seen'],
            'malware_type': family_info['type'],
            'alternatives': alternatives
        }

    def save(self, path: str) -> None:
        """Save the fine-tuned model."""
        os.makedirs(path, exist_ok=True)

        self.model.save_pretrained(os.path.join(path, 'model'))
        self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))

        metadata = {
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name,
            'families': self.families,
            'metrics': self.metrics,
            'history': self.history
        }

        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f"DistilBERT family classifier saved to {path}")

    def load(self, path: str) -> None:
        """Load a fine-tuned model."""
        from transformers import (
            DistilBertTokenizer,
            TFDistilBertForSequenceClassification
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            os.path.join(path, 'tokenizer')
        )
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            os.path.join(path, 'model')
        )

        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.max_length = metadata['max_length']
        self.families = metadata['families']
        self.metrics = metadata.get('metrics')

        print(f"DistilBERT family classifier loaded from {path}")
