"""
Training script for DGA Family Classification model.
Trains a model to classify DGA domains into specific malware families.
Used as a second-stage classifier after binary DGA detection.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.family_classifier import (
    DGAFamilyClassifier,
    LSTMFamilyClassifier,
    XGBoostFamilyClassifier,
    GradientBoostingFamilyClassifier,
    TransformerFamilyClassifier,
    DGA_FAMILY_INFO
)

# Optional: DistilBERT (requires transformers library)
try:
    from src.ml.family_classifier import DistilBERTFamilyClassifier
    DISTILBERT_AVAILABLE = True
except ImportError:
    DISTILBERT_AVAILABLE = False


def load_family_data(data_path: str, sample_per_family: int = None):
    """
    Load DGA family dataset.

    Args:
        data_path: Path to the dga_domains_full.csv
        sample_per_family: Optional sample size per family for faster testing

    Returns:
        Tuple of (domains list, families list)
    """
    print(f"Loading family dataset from {data_path}...")
    # Dataset has no header: class,family,domain
    df = pd.read_csv(data_path, header=None, names=['class', 'family', 'domain'])

    # Only use DGA domains (exclude legitimate/alexa)
    dga_df = df[df['class'] == 'dga'].copy()

    print(f"Total DGA domains: {len(dga_df)}")
    print(f"Families found: {dga_df['family'].nunique()}")

    # Show family distribution
    print("\nFamily distribution:")
    family_counts = dga_df['family'].value_counts()
    for family, count in family_counts.items():
        print(f"  {family}: {count}")

    # Optional sampling for faster testing
    if sample_per_family:
        print(f"\nSampling {sample_per_family} domains per family...")
        dga_df = dga_df.groupby('family').apply(
            lambda x: x.sample(n=min(sample_per_family, len(x)), random_state=42)
        ).reset_index(drop=True)
        print(f"Sampled {len(dga_df)} domains total")

    domains = dga_df['domain'].tolist()
    families = dga_df['family'].tolist()

    return domains, families


def plot_family_confusion_matrix(cm, families, title, save_path):
    """Plot and save confusion matrix for multi-class classification."""
    plt.figure(figsize=(16, 14))

    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=False,  # Too many classes for annotations
        cmap='Blues',
        xticklabels=families,
        yticklabels=families
    )
    plt.title(title)
    plt.xlabel('Predicted Family')
    plt.ylabel('Actual Family')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_family_metrics(classification_report, save_path):
    """Plot per-family F1 scores."""
    # Extract per-family metrics
    families = []
    f1_scores = []
    precisions = []
    recalls = []

    for family, metrics in classification_report.items():
        if family in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        families.append(family)
        f1_scores.append(metrics['f1-score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])

    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)[::-1]
    families = [families[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]

    x = np.arange(len(families))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width, precisions, width, label='Precision', color='#3b82f6')
    ax.bar(x, recalls, width, label='Recall', color='#22c55e')
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='#8b5cf6')

    ax.set_ylabel('Score')
    ax.set_title('Per-Family Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Family metrics plot saved to {save_path}")


def plot_family_distribution(families, save_path):
    """Plot family distribution in the dataset."""
    family_counts = pd.Series(families).value_counts()

    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(family_counts)))
    bars = plt.barh(range(len(family_counts)), family_counts.values, color=colors)
    plt.yticks(range(len(family_counts)), family_counts.index)
    plt.xlabel('Number of Domains')
    plt.title('DGA Family Distribution in Training Data')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Family distribution plot saved to {save_path}")


def plot_lstm_training_history(history, save_path):
    """Plot LSTM training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.get('loss', []), label='Train', color='#3b82f6')
    axes[0].plot(history.get('val_loss', []), label='Validation', color='#ef4444')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.get('accuracy', []), label='Train', color='#3b82f6')
    axes[1].plot(history.get('val_accuracy', []), label='Validation', color='#ef4444')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"LSTM training history saved to {save_path}")


def train_rf_family_classifier(
    domains, families, word_dict_path, legit_path,
    models_dir, plots_dir
):
    """Train Random Forest family classifier."""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST FAMILY CLASSIFIER")
    print("="*60)

    classifier = DGAFamilyClassifier(
        n_estimators=200,
        max_depth=25,
        word_dict_path=word_dict_path,
        legit_domains_path=legit_path,
        use_tfidf=True
    )

    X, y = classifier.prepare_data(domains, families)
    metrics = classifier.train(X, y, test_size=0.2)

    # Save model
    model_path = os.path.join(models_dir, 'family_classifier_rf.joblib')
    classifier.save(model_path)

    # Save plots
    plot_family_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        metrics['families'],
        'Random Forest Family Classifier - Confusion Matrix',
        os.path.join(plots_dir, 'family_rf_confusion_matrix.png')
    )

    if 'classification_report' in metrics:
        plot_family_metrics(
            metrics['classification_report'],
            os.path.join(plots_dir, 'family_rf_metrics.png')
        )

    # Feature importance
    importances = classifier.get_feature_importance()
    if importances:
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
        features, values = zip(*sorted_features)

        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        plt.barh(range(len(features)), values, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title('Top 20 Features for Family Classification')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'family_rf_feature_importance.png'), dpi=150)
        plt.close()

    return metrics


def train_lstm_family_classifier(
    domains, families, models_dir, plots_dir, epochs=30
):
    """Train LSTM family classifier."""
    print("\n" + "="*60)
    print("TRAINING LSTM FAMILY CLASSIFIER")
    print("="*60)

    classifier = LSTMFamilyClassifier(
        max_length=63,
        embedding_dim=64,
        lstm_units=128,
        dropout_rate=0.3
    )

    X, y = classifier.prepare_data(domains, families)
    metrics = classifier.train(X, y, test_size=0.2, epochs=epochs, batch_size=256)

    # Save model
    model_path = os.path.join(models_dir, 'family_classifier_lstm')
    classifier.save(model_path)

    # Save plots
    plot_family_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        metrics['families'],
        'LSTM Family Classifier - Confusion Matrix',
        os.path.join(plots_dir, 'family_lstm_confusion_matrix.png')
    )

    if 'history' in metrics:
        plot_lstm_training_history(
            metrics['history'],
            os.path.join(plots_dir, 'family_lstm_training_history.png')
        )

    return metrics


def train_xgboost_family_classifier(
    domains, families, word_dict_path, legit_path,
    models_dir, plots_dir
):
    """Train XGBoost family classifier."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST FAMILY CLASSIFIER")
    print("="*60)

    try:
        classifier = XGBoostFamilyClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            word_dict_path=word_dict_path,
            legit_domains_path=legit_path,
            use_tfidf=True
        )

        X, y = classifier.prepare_data(domains, families)
        metrics = classifier.train(X, y, test_size=0.2)

        # Save model
        model_path = os.path.join(models_dir, 'family_classifier_xgboost.joblib')
        classifier.save(model_path)

        # Save plots
        plot_family_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            metrics['families'],
            'XGBoost Family Classifier - Confusion Matrix',
            os.path.join(plots_dir, 'family_xgb_confusion_matrix.png')
        )

        # Feature importance
        importances = classifier.get_feature_importance()
        if importances:
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
            features, values = zip(*sorted_features)

            plt.figure(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
            plt.barh(range(len(features)), values, color=colors)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance')
            plt.title('XGBoost Family Classifier - Top 20 Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'family_xgb_feature_importance.png'), dpi=150)
            plt.close()
            print(f"XGBoost feature importance saved to {plots_dir}/family_xgb_feature_importance.png")

        # Save individual results JSON
        xgb_results = {
            'accuracy': metrics.get('accuracy'),
            'precision_micro': metrics.get('precision_micro'),
            'precision_macro': metrics.get('precision_macro'),
            'precision_weighted': metrics.get('precision_weighted'),
            'recall_micro': metrics.get('recall_micro'),
            'recall_macro': metrics.get('recall_macro'),
            'recall_weighted': metrics.get('recall_weighted'),
            'f1_micro': metrics.get('f1_micro'),
            'f1_macro': metrics.get('f1_macro'),
            'f1_weighted': metrics.get('f1_weighted'),
            'num_families': metrics.get('num_families'),
            'families': metrics.get('families')
        }
        with open(os.path.join(models_dir, 'family_xgb_results.json'), 'w') as f:
            json.dump(xgb_results, f, indent=2)
        print(f"XGBoost results saved to {models_dir}/family_xgb_results.json")

        return metrics

    except ImportError as e:
        print(f"XGBoost not available: {e}")
        return None


def train_gb_family_classifier(
    domains, families, word_dict_path, legit_path,
    models_dir, plots_dir
):
    """Train Gradient Boosting family classifier."""
    print("\n" + "="*60)
    print("TRAINING GRADIENT BOOSTING FAMILY CLASSIFIER")
    print("="*60)

    classifier = GradientBoostingFamilyClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        word_dict_path=word_dict_path,
        legit_domains_path=legit_path,
        use_tfidf=True
    )

    X, y = classifier.prepare_data(domains, families)
    metrics = classifier.train(X, y, test_size=0.2)

    # Save model
    model_path = os.path.join(models_dir, 'family_classifier_gb.joblib')
    classifier.save(model_path)

    # Save plots
    plot_family_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        metrics['families'],
        'Gradient Boosting Family Classifier - Confusion Matrix',
        os.path.join(plots_dir, 'family_gb_confusion_matrix.png')
    )

    # Feature importance
    importances = classifier.get_feature_importance()
    if importances:
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
        features, values = zip(*sorted_features)

        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        plt.barh(range(len(features)), values, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title('Gradient Boosting Family Classifier - Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'family_gb_feature_importance.png'), dpi=150)
        plt.close()
        print(f"GB feature importance saved to {plots_dir}/family_gb_feature_importance.png")

    # Save individual results JSON
    gb_results = {
        'accuracy': metrics.get('accuracy'),
        'precision_micro': metrics.get('precision_micro'),
        'precision_macro': metrics.get('precision_macro'),
        'precision_weighted': metrics.get('precision_weighted'),
        'recall_micro': metrics.get('recall_micro'),
        'recall_macro': metrics.get('recall_macro'),
        'recall_weighted': metrics.get('recall_weighted'),
        'f1_micro': metrics.get('f1_micro'),
        'f1_macro': metrics.get('f1_macro'),
        'f1_weighted': metrics.get('f1_weighted'),
        'num_families': metrics.get('num_families'),
        'families': metrics.get('families')
    }
    with open(os.path.join(models_dir, 'family_gb_results.json'), 'w') as f:
        json.dump(gb_results, f, indent=2)
    print(f"GB results saved to {models_dir}/family_gb_results.json")

    return metrics


def train_transformer_family_classifier(
    domains, families, models_dir, plots_dir, epochs=30
):
    """Train Transformer family classifier."""
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER FAMILY CLASSIFIER")
    print("="*60)

    classifier = TransformerFamilyClassifier(
        max_length=63,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout_rate=0.1
    )

    X, y = classifier.prepare_data(domains, families)
    metrics = classifier.train(X, y, test_size=0.2, epochs=epochs, batch_size=256)

    # Save model
    model_path = os.path.join(models_dir, 'family_classifier_transformer')
    classifier.save(model_path)

    # Save plots
    plot_family_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        metrics['families'],
        'Transformer Family Classifier - Confusion Matrix',
        os.path.join(plots_dir, 'family_transformer_confusion_matrix.png')
    )

    if 'history' in metrics:
        plot_lstm_training_history(
            metrics['history'],
            os.path.join(plots_dir, 'family_transformer_training_history.png')
        )

    return metrics


def train_distilbert_family_classifier(
    domains, families, models_dir, plots_dir, epochs=3
):
    """Train DistilBERT family classifier."""
    print("\n" + "="*60)
    print("TRAINING DISTILBERT FAMILY CLASSIFIER")
    print("="*60)

    if not DISTILBERT_AVAILABLE:
        print("DistilBERT not available. Install transformers: pip install transformers")
        return None

    classifier = DistilBERTFamilyClassifier(
        max_length=128,
        learning_rate=2e-5
    )

    X, y = classifier.prepare_data(domains, families)
    metrics = classifier.train(X, y, test_size=0.2, epochs=epochs, batch_size=32)

    # Save model
    model_path = os.path.join(models_dir, 'family_classifier_distilbert')
    classifier.save(model_path)

    # Save plots
    plot_family_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        metrics['families'],
        'DistilBERT Family Classifier - Confusion Matrix',
        os.path.join(plots_dir, 'family_distilbert_confusion_matrix.png')
    )

    if 'history' in metrics:
        plot_lstm_training_history(
            metrics['history'],
            os.path.join(plots_dir, 'family_distilbert_training_history.png')
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train DGA Family Classification model')
    parser.add_argument('--data-path', type=str, default='data/raw/dga_domains_full.csv',
                       help='Path to family-labeled DGA dataset')
    parser.add_argument('--word-dict', type=str, default='data/raw/words.txt',
                       help='Path to word dictionary')
    parser.add_argument('--legit-path', type=str, default='data/raw/legit_websites.csv',
                       help='Path to legitimate domains (for TF-IDF)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--plots-dir', type=str, default='models/plots',
                       help='Directory to save plots')
    parser.add_argument('--sample-per-family', type=int, default=None,
                       help='Sample size per family (for faster testing)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs for LSTM training')
    parser.add_argument('--rf-only', action='store_true',
                       help='Train only Random Forest model')
    parser.add_argument('--lstm-only', action='store_true',
                       help='Train only LSTM model')
    parser.add_argument('--xgboost-only', action='store_true',
                       help='Train only XGBoost model')
    parser.add_argument('--gb-only', action='store_true',
                       help='Train only Gradient Boosting model')
    parser.add_argument('--transformer-only', action='store_true',
                       help='Train only Transformer model')
    parser.add_argument('--distilbert-only', action='store_true',
                       help='Train only DistilBERT model')
    parser.add_argument('--all-models', action='store_true',
                       help='Train all models including XGBoost, GB, Transformer, and DistilBERT')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Load data
    domains, families = load_family_data(args.data_path, args.sample_per_family)

    # Plot family distribution
    plot_family_distribution(
        families,
        os.path.join(args.plots_dir, 'family_distribution.png')
    )

    results = {}

    # Determine which models to train
    train_specific = (args.rf_only or args.lstm_only or args.xgboost_only or
                      args.gb_only or args.transformer_only or args.distilbert_only)

    # Train Random Forest
    if args.rf_only or (not train_specific):
        rf_metrics = train_rf_family_classifier(
            domains, families,
            args.word_dict,
            args.legit_path,
            args.models_dir,
            args.plots_dir
        )
        results['Random Forest'] = rf_metrics

    # Train LSTM
    if args.lstm_only or (not train_specific):
        lstm_metrics = train_lstm_family_classifier(
            domains, families,
            args.models_dir,
            args.plots_dir,
            args.epochs
        )
        results['LSTM'] = lstm_metrics

    # Train XGBoost
    if args.xgboost_only or args.all_models:
        xgb_metrics = train_xgboost_family_classifier(
            domains, families,
            args.word_dict,
            args.legit_path,
            args.models_dir,
            args.plots_dir
        )
        if xgb_metrics:
            results['XGBoost'] = xgb_metrics

    # Train Gradient Boosting
    if args.gb_only or args.all_models:
        gb_metrics = train_gb_family_classifier(
            domains, families,
            args.word_dict,
            args.legit_path,
            args.models_dir,
            args.plots_dir
        )
        results['Gradient Boosting'] = gb_metrics

    # Train Transformer
    if args.transformer_only or args.all_models:
        transformer_metrics = train_transformer_family_classifier(
            domains, families,
            args.models_dir,
            args.plots_dir,
            args.epochs
        )
        results['Transformer'] = transformer_metrics

    # Train DistilBERT
    if args.distilbert_only or args.all_models:
        distilbert_metrics = train_distilbert_family_classifier(
            domains, families,
            args.models_dir,
            args.plots_dir,
            epochs=3
        )
        if distilbert_metrics:
            results['DistilBERT'] = distilbert_metrics

    # Compare models
    if len(results) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*80)

        print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 (micro)':<12} {'F1 (macro)':<12} {'F1 (weighted)':<12}")
        print("-" * 68)

        for model_name, metrics in results.items():
            acc = metrics.get('accuracy', 0)
            f1_micro = metrics.get('f1_micro', 0)
            f1_macro = metrics.get('f1_macro', 0)
            f1_weighted = metrics.get('f1_weighted', f1_macro)
            print(f"{model_name:<20} {acc:<12.4f} {f1_micro:<12.4f} {f1_macro:<12.4f} {f1_weighted:<12.4f}")

    # Save results summary
    results_summary = {
        model: {
            'accuracy': metrics.get('accuracy'),
            'precision_micro': metrics.get('precision_micro'),
            'precision_macro': metrics.get('precision_macro'),
            'precision_weighted': metrics.get('precision_weighted'),
            'recall_micro': metrics.get('recall_micro'),
            'recall_macro': metrics.get('recall_macro'),
            'recall_weighted': metrics.get('recall_weighted'),
            'f1_micro': metrics.get('f1_micro'),
            'f1_macro': metrics.get('f1_macro'),
            'f1_weighted': metrics.get('f1_weighted'),
            'num_families': metrics.get('num_families'),
            'families': metrics.get('families')
        }
        for model, metrics in results.items()
    }

    with open(os.path.join(args.models_dir, 'family_classifier_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    # Print family info
    print("\n" + "="*60)
    print("DGA FAMILY THREAT INTELLIGENCE")
    print("="*60)

    unique_families = set(families)
    threat_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}

    for family in unique_families:
        info = DGA_FAMILY_INFO.get(family, {'threat_level': 'unknown'})
        threat_counts[info.get('threat_level', 'unknown')] += 1

    print(f"\nThreat Level Distribution:")
    for level, count in threat_counts.items():
        if count > 0:
            print(f"  {level.capitalize()}: {count} families")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {args.models_dir}")
    print(f"Plots saved to: {args.plots_dir}")
    print(f"\nTo use the family classifier:")
    print("  1. First detect if domain is DGA using binary classifier")
    print("  2. If DGA detected, use family classifier to identify malware family")


if __name__ == '__main__':
    main()
