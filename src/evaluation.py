import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, classification_report)
import glob

def evaluate_model(model, X_test, y_test, model_name, metrics_path, cm_path, threshold=0.5):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'model': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print(f"\n{'─'*50}")
    print(f"Model: {model_name}")
    for k, v in metrics.items():
        if k != 'model':
            print(f"{k:10s}: {v:.4f}")

    print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)

    return metrics

def plot_history(history, save_path=None):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    if save_path:
        plt.savefig(save_path)


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Phishing'], yticklabels=['Safe', 'Phishing'], ax=ax)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_bar(metrics_list, save_path):
    if not metrics_list:
        return

    model_names = [m['model'] for m in metrics_list]
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 2), 5))

    for i, key in enumerate(metric_keys):
        vals = [m[key] for m in metrics_list]
        bars = ax.bar(x + i * width, vals, width, label=key.capitalize())
        ax.bar_label(bars, fmt='%.3f', fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=25)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_and_compare_all_metrics(results_dir = 'results'):
    files = sorted(glob.glob(os.path.join(results_dir, '*_metrics.json')))
    if not files:
        print("No metrics JSON files in:", results_dir)
        return

    all_metrics = []
    for f in files:
        with open(f) as fh:
            all_metrics.append(json.load(fh))

    plot_comparison_bar(all_metrics, results_dir)
    return all_metrics

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res  = os.path.join(base, 'results')
    load_and_compare_all_metrics(res)