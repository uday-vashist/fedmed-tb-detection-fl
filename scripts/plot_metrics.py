"""
Generates performance plots for FedMed: a grouped bar chart comparing
Accuracy / Precision / Recall / Specificity / F1 between the centralized
baseline and federated models, plus confusion matrix heatmaps for each.

Reuses the model-loading and metric-computation logic from evaluate_metrics.py
so there is a single source of truth for how metrics are calculated.

Usage:
    python scripts/plot_metrics.py

Requires (in addition to evaluate_metrics.py's requirements):
    pip install matplotlib
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
if os.path.basename(_THIS_DIR) == "scripts":
    _PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save files
import matplotlib.pyplot as plt
import numpy as np

from evaluate_metrics import (
    get_fixed_test_loader, run_inference, compute_metrics,
    get_model, DEVICE, MODELS_DIR
)
import torch

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "plots")


def plot_grouped_bar(results, output_path):
    """Grouped bar chart: 5 metrics x 2 models."""
    metrics_order = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall\n(Sensitivity)"),
        ("specificity", "Specificity"),
        ("f1", "F1 Score"),
    ]
    labels = [m[1] for m in metrics_order]
    baseline_vals = [results[0][m[0]] * 100 for m in metrics_order]
    federated_vals = [results[1][m[0]] * 100 for m in metrics_order]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Centralized Baseline", color="#2E86AB")
    bars2 = ax.bar(x + width/2, federated_vals, width, label="Federated (FedAvg)", color="#E67E22")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance Comparison: Baseline vs Federated", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrix(result, model_name, output_path):
    """Confusion matrix heatmap for a single model."""
    cm = np.array([[result["tn"], result["fp"]],
                   [result["fn"], result["tp"]]])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Healthy", "TB"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = DEVICE
    print(f"Using device: {device}")

    print("\nBuilding fixed-seed test set...")
    test_loader = get_fixed_test_loader()

    results = []

    print("\nLoading centralized baseline model...")
    baseline_model = get_model(pretrained=True, freeze_backbone=False).to(device)
    baseline_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'baseline_best.pth'), map_location=device)
    )
    y_true, y_pred = run_inference(baseline_model, test_loader, device)
    results.append(compute_metrics(y_true, y_pred, "Centralized Baseline"))

    print("\nLoading federated global model...")
    fl_model = get_model(pretrained=True, freeze_backbone=False).to(device)
    fl_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'fl_best.pth'), map_location=device)
    )
    y_true, y_pred = run_inference(fl_model, test_loader, device)
    results.append(compute_metrics(y_true, y_pred, "Federated (FedAvg) Model"))

    print("\nGenerating plots...")
    plot_grouped_bar(results, os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
    plot_confusion_matrix(results[0], "Centralized Baseline", os.path.join(OUTPUT_DIR, "confusion_matrix_baseline.png"))
    plot_confusion_matrix(results[1], "Federated Model", os.path.join(OUTPUT_DIR, "confusion_matrix_federated.png"))

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()  