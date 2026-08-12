"""
Evaluation script for FedMed: computes precision, recall, F1, sensitivity,
specificity, and confusion matrix for both the centralized baseline model
and the federated global model, using the saved .pth checkpoints.

Usage:
    python evaluate_metrics.py

Requires:
    - src/model.py, src/data_loader.py, src/config.py (unchanged, already in repo)
    - models/baseline_best.pth
    - models/fl_best.pth  (rename here if your FL checkpoint has a different filename)
    - sklearn installed (pip install scikit-learn)
"""

import os
import sys

# Make sure the project root (the folder containing "src/") is on the
# import path, regardless of whether this script is run from the project
# root or from inside scripts/. This means you don't need to worry about
# how you normally invoke train.py — this works either way.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _THIS_DIR
if os.path.basename(_THIS_DIR) == "scripts":
    _PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)

from src.config import *
from src.model import get_model
from src.data_loader import get_data_loaders

# Fixed seed so this test split is reproducible for future re-runs /
# sanity checks. Note: this will NOT be pixel-identical to whichever
# unseeded split produced the test accuracy already reported in the
# manuscript, since no seed was saved during original training.
EVAL_SEED = 42


def get_fixed_test_loader(batch_size=BATCH_SIZE):
    """Rebuild data loaders with a fixed seed for reproducible evaluation."""
    from src.data_loader import load_combined_data, TBDataset, get_transforms
    from torch.utils.data import DataLoader

    _, _, test_data = load_combined_data(random_seed=EVAL_SEED)
    test_loader = DataLoader(
        TBDataset(test_data[0], test_data[1], get_transforms(train=False)),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return test_loader


def run_inference(model, loader, device):
    """Run model over loader, return (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred, model_name):
    """
    Compute and print precision, recall, F1, sensitivity, specificity.
    Positive class = 1 (TB). Confusion matrix layout:
        [[TN, FP],
         [FN, TP]]
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # = sensitivity
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    sensitivity = recall  # same metric, medical naming
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("\n" + "=" * 60)
    print(f"{model_name} — Test Set Evaluation")
    print("=" * 60)
    print(f"Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Pred Healthy   Pred TB")
    print(f"  Actual Healthy      {tn:5d}       {fp:5d}")
    print(f"  Actual TB           {fn:5d}       {tp:5d}")
    print()
    print(f"Accuracy    : {accuracy*100:.2f}%")
    print(f"Precision   : {precision*100:.2f}%")
    print(f"Recall      : {recall*100:.2f}%")
    print(f"Sensitivity : {sensitivity*100:.2f}%   (= Recall, TB detection rate)")
    print(f"Specificity : {specificity*100:.2f}%   (Healthy correctly cleared)")
    print(f"F1 Score    : {f1*100:.2f}%")
    print("=" * 60)

    return {
        "model": model_name,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "sensitivity": sensitivity, "specificity": specificity, "f1": f1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }


def main():
    device = DEVICE
    print(f"Using device: {device}")

    print("\nBuilding fixed-seed test set (seed={})...".format(EVAL_SEED))
    test_loader = get_fixed_test_loader()

    results = []

    # ---- Centralized baseline ----
    print("\nLoading centralized baseline model...")
    baseline_model = get_model(pretrained=True, freeze_backbone=False).to(device)
    baseline_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'baseline_best.pth'), map_location=device)
    )
    y_true, y_pred = run_inference(baseline_model, test_loader, device)
    results.append(compute_metrics(y_true, y_pred, "Centralized Baseline"))

    # ---- Federated global model ----
    print("\nLoading federated global model...")
    fl_model = get_model(pretrained=True, freeze_backbone=False).to(device)
    fl_model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'fl_best.pth'), map_location=device)
    )
    y_true, y_pred = run_inference(fl_model, test_loader, device)
    results.append(compute_metrics(y_true, y_pred, "Federated (FedAvg) Model"))

    # ---- Summary table for the paper ----
    print("\n\n" + "=" * 60)
    print("SUMMARY TABLE (copy into manuscript)")
    print("=" * 60)
    print(f"{'Metric':<15}{'Baseline':<15}{'Federated':<15}")
    for key, label in [("accuracy", "Accuracy"), ("precision", "Precision"),
                        ("recall", "Recall/Sens."), ("specificity", "Specificity"),
                        ("f1", "F1 Score")]:
        print(f"{label:<15}{results[0][key]*100:<15.2f}{results[1][key]*100:<15.2f}")

    return results


if __name__ == "__main__":
    main()