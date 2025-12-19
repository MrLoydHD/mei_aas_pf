"""
Training script for DGA detection models.
Trains Random Forest, LSTM, and optionally XGBoost/Gradient Boosting models.
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

from src.ml.features import DomainFeatureExtractor, remove_outliers
from src.ml.random_forest_model import RandomForestDGADetector
from src.ml.lstm_model import LSTMDGADetector
from src.ml.xgboost_model import XGBoostDGADetector
from src.ml.gradient_boosting_model import GradientBoostingDGADetector
from src.ml.transformer_model import TransformerDGADetector

# Optional: DistilBERT (requires transformers library)
try:
    from src.ml.distilbert_model import DistilBERTDGADetector
    DISTILBERT_AVAILABLE = True
except ImportError:
    DISTILBERT_AVAILABLE = False


def load_data(dga_path: str, legit_path: str, sample_size: int = None):
    """Load and optionally sample domain data."""
    print(f"Loading DGA domains from {dga_path}...")
    dga_df = pd.read_csv(dga_path)
    dga_df = dga_df.dropna().drop_duplicates()

    print(f"Loading legitimate domains from {legit_path}...")
    legit_df = pd.read_csv(legit_path)
    legit_df = legit_df.dropna().drop_duplicates()

    if sample_size:
        print(f"Sampling {sample_size} domains from each class...")
        dga_df = dga_df.sample(n=min(sample_size, len(dga_df)), random_state=42)
        legit_df = legit_df.sample(n=min(sample_size, len(legit_df)), random_state=42)

    dga_domains = dga_df['domain'].tolist()
    legit_domains = legit_df['domain'].tolist()

    print(f"Loaded {len(dga_domains)} DGA domains and {len(legit_domains)} legitimate domains")

    return dga_domains, legit_domains


def plot_confusion_matrix(cm, title, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'DGA'],
                yticklabels=['Legitimate', 'DGA'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(importances, save_path):
    """Plot and save feature importance."""
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_features)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
    bars = plt.barh(range(len(features)), values, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history, save_path):
    """Plot LSTM training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history.get('loss', []), label='Train')
    axes[0].plot(history.get('val_loss', []), label='Validation')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.get('accuracy', []), label='Train')
    axes[1].plot(history.get('val_accuracy', []), label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    # AUC
    axes[2].plot(history.get('auc', []), label='Train')
    axes[2].plot(history.get('val_auc', []), label='Validation')
    axes[2].set_title('AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(all_metrics, save_path):
    """Plot comparison of all models."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(all_metrics.keys())

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        bars = ax.bar(x + i * width - width * len(model_names) / 2, values, width,
                      label=model_name, color=colors[i])

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_random_forest(dga_domains, legit_domains, word_dict_path, legit_path, models_dir, plots_dir):
    """Train Random Forest model with TF-IDF features."""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)

    rf_model = RandomForestDGADetector(
        n_estimators=200,
        max_depth=20,
        word_dict_path=word_dict_path,
        legit_domains_path=legit_path,
        use_tfidf=True
    )

    X, y = rf_model.prepare_data(dga_domains, legit_domains)
    metrics = rf_model.train(X, y, test_size=0.2)

    # Save model
    rf_model.save(os.path.join(models_dir, 'random_forest.joblib'))

    # Save plots
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        'Random Forest Confusion Matrix',
        os.path.join(plots_dir, 'rf_confusion_matrix.png')
    )

    plot_feature_importance(
        rf_model.get_feature_importance(),
        os.path.join(plots_dir, 'rf_feature_importance.png')
    )

    return metrics


def train_xgboost(dga_domains, legit_domains, word_dict_path, legit_path, models_dir, plots_dir):
    """Train XGBoost model using XGBoostDGADetector class."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)

    try:
        xgb_model = XGBoostDGADetector(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            word_dict_path=word_dict_path,
            legit_domains_path=legit_path,
            use_tfidf=True
        )

        X, y = xgb_model.prepare_data(dga_domains, legit_domains)
        metrics = xgb_model.train(X, y, test_size=0.2)

        # Save model
        xgb_model.save(os.path.join(models_dir, 'xgboost.joblib'))

        # Save plots
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            'XGBoost Confusion Matrix',
            os.path.join(plots_dir, 'xgb_confusion_matrix.png')
        )

        plot_feature_importance(
            xgb_model.get_feature_importance(),
            os.path.join(plots_dir, 'xgb_feature_importance.png')
        )

        return metrics

    except ImportError as e:
        print(f"XGBoost not installed. Skipping XGBoost training: {e}")
        print("Install with: pip install xgboost")
        return None


def train_gradient_boosting(dga_domains, legit_domains, word_dict_path, legit_path, models_dir, plots_dir):
    """Train Gradient Boosting model using GradientBoostingDGADetector class."""
    print("\n" + "="*60)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("="*60)

    gb_model = GradientBoostingDGADetector(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        word_dict_path=word_dict_path,
        legit_domains_path=legit_path,
        use_tfidf=True
    )

    X, y = gb_model.prepare_data(dga_domains, legit_domains)
    metrics = gb_model.train(X, y, test_size=0.2)

    # Save model
    gb_model.save(os.path.join(models_dir, 'gradient_boosting.joblib'))

    # Save plots
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        'Gradient Boosting Confusion Matrix',
        os.path.join(plots_dir, 'gb_confusion_matrix.png')
    )

    plot_feature_importance(
        gb_model.get_feature_importance(),
        os.path.join(plots_dir, 'gb_feature_importance.png')
    )

    return metrics


def train_transformer(dga_domains, legit_domains, models_dir, plots_dir, epochs=30):
    """Train Transformer model using TransformerDGADetector class."""
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER MODEL")
    print("="*60)

    transformer_model = TransformerDGADetector(
        max_length=63,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout_rate=0.1
    )

    X, y = transformer_model.prepare_data(dga_domains, legit_domains)
    metrics = transformer_model.train(
        X, y,
        test_size=0.2,
        epochs=epochs,
        batch_size=256
    )

    # Save model
    transformer_model.save(os.path.join(models_dir, 'transformer'))

    # Save plots
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        'Transformer Confusion Matrix',
        os.path.join(plots_dir, 'transformer_confusion_matrix.png')
    )

    history = transformer_model.get_training_history()
    if history:
        plot_training_history(history, os.path.join(plots_dir, 'transformer_training_history.png'))

    return metrics


def train_distilbert(dga_domains, legit_domains, models_dir, plots_dir, epochs=3):
    """Train DistilBERT model using DistilBERTDGADetector class."""
    print("\n" + "="*60)
    print("TRAINING DISTILBERT MODEL")
    print("="*60)

    if not DISTILBERT_AVAILABLE:
        print("DistilBERT not available. Install transformers: pip install transformers")
        return None

    distilbert_model = DistilBERTDGADetector(
        max_length=128,
        learning_rate=2e-5
    )

    X, y = distilbert_model.prepare_data(dga_domains, legit_domains)
    metrics = distilbert_model.train(
        X, y,
        test_size=0.2,
        epochs=epochs,
        batch_size=32
    )

    # Save model
    distilbert_model.save(os.path.join(models_dir, 'distilbert'))

    # Save plots
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        'DistilBERT Confusion Matrix',
        os.path.join(plots_dir, 'distilbert_confusion_matrix.png')
    )

    history = distilbert_model.get_training_history()
    if history:
        plot_training_history(history, os.path.join(plots_dir, 'distilbert_training_history.png'))

    return metrics


def train_lstm(dga_domains, legit_domains, models_dir, plots_dir, epochs=30):
    """Train LSTM model."""
    print("\n" + "="*60)
    print("TRAINING LSTM MODEL")
    print("="*60)

    lstm_model = LSTMDGADetector(
        max_length=63,
        embedding_dim=64,
        lstm_units=128,
        dropout_rate=0.3
    )

    X, y = lstm_model.prepare_data(dga_domains, legit_domains)
    metrics = lstm_model.train(
        X, y,
        test_size=0.2,
        epochs=epochs,
        batch_size=256,
        use_cnn_lstm=True
    )

    # Save model
    lstm_model.save(os.path.join(models_dir, 'lstm'))

    # Save plots
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        'LSTM Confusion Matrix',
        os.path.join(plots_dir, 'lstm_confusion_matrix.png')
    )

    history = lstm_model.get_training_history()
    if history:
        plot_training_history(history, os.path.join(plots_dir, 'lstm_training_history.png'))

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train DGA detection models')
    parser.add_argument('--dga-path', type=str, default='data/raw/dga_websites.csv',
                       help='Path to DGA domains CSV')
    parser.add_argument('--legit-path', type=str, default='data/raw/legit_websites.csv',
                       help='Path to legitimate domains CSV')
    parser.add_argument('--word-dict', type=str, default='data/raw/words.txt',
                       help='Path to word dictionary')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--plots-dir', type=str, default='models/plots',
                       help='Directory to save plots')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for training (use for faster testing)')
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
    dga_domains, legit_domains = load_data(
        args.dga_path,
        args.legit_path,
        args.sample_size
    )

    results = {}

    # Determine which models to train
    train_specific = (args.rf_only or args.lstm_only or args.xgboost_only or
                      args.gb_only or args.transformer_only or args.distilbert_only)

    # Train Random Forest
    if args.rf_only or (not train_specific):
        rf_metrics = train_random_forest(
            dga_domains, legit_domains,
            args.word_dict,
            args.legit_path,
            args.models_dir,
            args.plots_dir
        )
        results['Random Forest'] = rf_metrics

    # Train LSTM
    if args.lstm_only or (not train_specific):
        lstm_metrics = train_lstm(
            dga_domains, legit_domains,
            args.models_dir,
            args.plots_dir,
            args.epochs
        )
        results['LSTM'] = lstm_metrics

    # Train XGBoost
    if args.xgboost_only or args.all_models:
        xgb_metrics = train_xgboost(
            dga_domains, legit_domains,
            args.word_dict,
            args.legit_path,
            args.models_dir,
            args.plots_dir
        )
        if xgb_metrics:
            results['XGBoost'] = xgb_metrics

    # Train Gradient Boosting
    if args.gb_only or args.all_models:
        gb_metrics = train_gradient_boosting(
            dga_domains, legit_domains,
            args.word_dict,
            args.legit_path,
            args.models_dir,
            args.plots_dir
        )
        results['Gradient Boosting'] = gb_metrics

    # Train Transformer
    if args.transformer_only or args.all_models:
        transformer_metrics = train_transformer(
            dga_domains, legit_domains,
            args.models_dir,
            args.plots_dir,
            args.epochs
        )
        results['Transformer'] = transformer_metrics

    # Train DistilBERT
    if args.distilbert_only or args.all_models:
        distilbert_metrics = train_distilbert(
            dga_domains, legit_domains,
            args.models_dir,
            args.plots_dir,
            epochs=3  # DistilBERT typically needs fewer epochs
        )
        if distilbert_metrics:
            results['DistilBERT'] = distilbert_metrics

    # Plot comparison if multiple models trained
    if len(results) > 1:
        plot_comparison(
            results,
            os.path.join(args.plots_dir, 'model_comparison.png')
        )

    # Save results summary
    results_summary = {
        model: {k: v for k, v in metrics.items()
                if k not in ['confusion_matrix', 'classification_report']}
        for model, metrics in results.items()
    }

    with open(os.path.join(args.models_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {args.models_dir}")
    print(f"Plots saved to: {args.plots_dir}")

    if len(results) > 1:
        print("\n--- COMPARISON ---")
        header = f"{'Metric':<15}"
        for model in results.keys():
            header += f" {model:<15}"
        print(header)
        print("-" * (15 + 16 * len(results)))

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            row = f"{metric:<15}"
            for model, model_metrics in results.items():
                val = model_metrics.get(metric, 0)
                row += f" {val:<15.4f}"
            print(row)


if __name__ == '__main__':
    main()
