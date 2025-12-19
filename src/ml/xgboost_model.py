"""
XGBoost model for DGA detection.
Uses handcrafted features for classification.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split, cross_val_score

from .features import DomainFeatureExtractor


class XGBoostDGADetector:
    """
    XGBoost-based DGA detector.

    This model uses handcrafted statistical and linguistic features
    extracted from domain names to classify them as legitimate or DGA.
    XGBoost typically outperforms Random Forest with better handling
    of feature interactions and gradient boosting optimization.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        word_dict_path: Optional[str] = None,
        legit_domains_path: Optional[str] = None,
        use_tfidf: bool = True
    ):
        """
        Initialize the XGBoost DGA detector.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Boosting learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns for each tree
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for a split
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed for reproducibility
            word_dict_path: Path to word dictionary file
            legit_domains_path: Path to legitimate domains CSV for TF-IDF
            use_tfidf: Whether to use TF-IDF features
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=1  # Will be computed if data is imbalanced
        )

        self.feature_extractor = DomainFeatureExtractor(
            word_dict_path=word_dict_path,
            legit_domains_path=legit_domains_path,
            use_tfidf=use_tfidf
        )
        self.is_trained = False
        self.feature_names = self.feature_extractor.get_feature_names()
        self.metrics: Dict[str, Any] = {}

    def prepare_data(
        self,
        dga_domains: List[str],
        legit_domains: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and labels from domain lists.

        Args:
            dga_domains: List of DGA domain names
            legit_domains: List of legitimate domain names

        Returns:
            Tuple of (features, labels)
        """
        print(f"Extracting features from {len(dga_domains)} DGA domains...")
        dga_features = self.feature_extractor.extract_features_batch(dga_domains)

        print(f"Extracting features from {len(legit_domains)} legitimate domains...")
        legit_features = self.feature_extractor.extract_features_batch(legit_domains)

        # Combine and create labels (1 = DGA, 0 = Legitimate)
        X = pd.concat([dga_features, legit_features], ignore_index=True)
        y = np.array([1] * len(dga_domains) + [0] * len(legit_domains))

        return X.values, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validate: bool = True,
        early_stopping_rounds: Optional[int] = 10
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.

        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of data for testing
            validate: Whether to perform validation
            early_stopping_rounds: Rounds for early stopping (None to disable)

        Returns:
            Dictionary of training metrics
        """
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Compute scale_pos_weight for imbalanced data
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        if pos_count > 0:
            self.model.set_params(scale_pos_weight=neg_count / pos_count)

        print(f"Training on {len(X_train)} samples...")

        if early_stopping_rounds:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        self.is_trained = True

        if validate:
            print("Evaluating model...")
            self.metrics = self.evaluate(X_test, y_test)

            # Cross-validation
            print("Performing cross-validation...")
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            self.metrics['cv_accuracy_mean'] = cv_scores.mean()
            self.metrics['cv_accuracy_std'] = cv_scores.std()

            print(f"\nResults:")
            print(f"  Accuracy: {self.metrics['accuracy']:.4f}")
            print(f"  Precision: {self.metrics['precision']:.4f}")
            print(f"  Recall: {self.metrics['recall']:.4f}")
            print(f"  F1-Score: {self.metrics['f1']:.4f}")
            print(f"  ROC-AUC: {self.metrics['roc_auc']:.4f}")
            print(f"  CV Accuracy: {self.metrics['cv_accuracy_mean']:.4f} (+/- {self.metrics['cv_accuracy_std']:.4f})")

        return self.metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]

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

        features = self.feature_extractor.extract_features(domain)
        feature_vector = np.array([list(features.values())])

        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]

        return {
            'domain': domain,
            'is_dga': bool(prediction),
            'confidence': float(probabilities[prediction]),
            'dga_probability': float(probabilities[1]),
            'legit_probability': float(probabilities[0]),
            'features': features
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

        features_df = self.feature_extractor.extract_features_batch(domains)
        predictions = self.model.predict(features_df.values)
        probabilities = self.model.predict_proba(features_df.values)

        results = []
        for i, domain in enumerate(domains):
            results.append({
                'domain': domain,
                'is_dga': bool(predictions[i]),
                'confidence': float(probabilities[i][predictions[i]]),
                'dga_probability': float(probabilities[i][1]),
                'legit_probability': float(probabilities[i][0])
            })

        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'word_dict': self.feature_extractor.word_dict,
            'use_tfidf': self.feature_extractor.use_tfidf,
            'legit_tfidf_vectorizer': self.feature_extractor.legit_tfidf_vectorizer,
            'legit_tfidf_weights': self.feature_extractor.legit_tfidf_weights,
            'word_tfidf_vectorizer': self.feature_extractor.word_tfidf_vectorizer,
            'word_tfidf_weights': self.feature_extractor.word_tfidf_weights,
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.feature_extractor.word_dict = model_data.get('word_dict', set())
        self.feature_extractor.use_tfidf = model_data.get('use_tfidf', False)
        self.feature_extractor.legit_tfidf_vectorizer = model_data.get('legit_tfidf_vectorizer')
        self.feature_extractor.legit_tfidf_weights = model_data.get('legit_tfidf_weights')
        self.feature_extractor.word_tfidf_vectorizer = model_data.get('word_tfidf_vectorizer')
        self.feature_extractor.word_tfidf_weights = model_data.get('word_tfidf_weights')
        self.is_trained = True

        print(f"Model loaded from {path}")
