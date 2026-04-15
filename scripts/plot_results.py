# scripts/plot_results.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import METRICS_DIR, PLOTS_DIR

sns.set_style("whitegrid")


def plot_baseline(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'],   'r-s', label='Val Loss',   linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Baseline — Loss Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_acc'],   'r-s', label='Val Acc',   linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Baseline — Accuracy Curves'); ax2.legend(); ax2.grid(True, alpha=0.3)

    best_acc = max(history['val_acc'])
    best_ep  = history['val_acc'].index(best_acc) + 1
    ax2.annotate(f'Best: {best_acc:.2f}%\nEp {best_ep}',
                 xy=(best_ep, best_acc),
                 xytext=(best_ep + 0.5, best_acc - 5),
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7),
                 arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'baseline_curves.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out}")
    plt.close()


def plot_fl(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rounds = history['round']

    ax1.plot(rounds, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('FL Round'); ax1.set_ylabel('Loss')
    ax1.set_title('FL — Validation Loss per Round'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(rounds, history['avg_train_acc'], 'b-o', label='Avg Train Acc', linewidth=2, markersize=6)
    ax2.plot(rounds, history['val_acc'],       'r-s', label='Val Acc',       linewidth=2, markersize=6)
    ax2.set_xlabel('FL Round'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('FL — Accuracy per Round'); ax2.legend(); ax2.grid(True, alpha=0.3)

    best_acc   = max(history['val_acc'])
    best_round = history['val_acc'].index(best_acc) + 1
    ax2.annotate(f'Best: {best_acc:.2f}%\nRound {best_round}',
                 xy=(best_round, best_acc),
                 xytext=(best_round + 0.3, best_acc - 5),
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7),
                 arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'fl_curves.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out}")
    plt.close()


def plot_comparison(baseline, fl):
    fig, ax = plt.subplots(figsize=(9, 6))

    categories  = ['Baseline\n(Centralized)', 'Federated Learning\n(Best Round)']
    val_accs    = [max(baseline['val_acc']), max(fl['val_acc'])]
    colors      = ['#3498db', '#e67e22']

    bars = ax.bar(categories, val_accs, color=colors, alpha=0.8, edgecolor='black', width=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f'{h:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Accuracy (%)'); ax.set_title('Baseline vs Federated Learning — Val Accuracy')
    ax.set_ylim(0, 105)
    ax.axhline(y=85, color='green', linestyle='--', linewidth=1.5, label='Target: 85%')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'baseline_vs_fl.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {out}")
    plt.close()


def generate_report(baseline, fl):
    report_path = os.path.join(METRICS_DIR, 'training_report.txt')
    gap = baseline['train_acc'][-1] - baseline['val_acc'][-1]
    last3_std = (sum((x - sum(baseline['val_acc'][-3:])/3)**2
                     for x in baseline['val_acc'][-3:]) ** 0.5)

    best_fl_acc   = max(fl['val_acc'])
    best_fl_round = fl['val_acc'].index(best_fl_acc) + 1

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n TRAINING REPORT\n" + "="*60 + "\n\n")

        f.write("--- BASELINE (Centralized) ---\n")
        f.write(f"Total epochs       : {len(baseline['train_loss'])}\n")
        f.write(f"Final train acc    : {baseline['train_acc'][-1]:.2f}%\n")
        f.write(f"Final val acc      : {baseline['val_acc'][-1]:.2f}%\n")
        f.write(f"Best val acc       : {max(baseline['val_acc']):.2f}%\n")
        f.write(f"Train-Val gap      : {gap:.2f}% ")
        f.write("Good\n" if gap < 5 else "Moderate overfit\n" if gap < 10 else "Overfit\n")
        f.write(f"Last 3 epochs std  : {last3_std:.2f}% ")
        f.write("Converged\n" if last3_std < 1 else "Still improving\n")

        f.write("\n--- FEDERATED LEARNING ---\n")
        f.write(f"Total rounds       : {len(fl['round'])}\n")
        f.write(f"Best val acc       : {best_fl_acc:.2f}% (round {best_fl_round})\n")
        f.write(f"Final val acc      : {fl['val_acc'][-1]:.2f}%\n")
        f.write(f"Final train acc    : {fl['avg_train_acc'][-1]:.2f}%\n")

        f.write("\n--- COMPARISON ---\n")
        diff = max(baseline['val_acc']) - best_fl_acc
        f.write(f"Accuracy gap (Baseline - FL): {diff:.2f}%\n")
        f.write("(Expected: FL trades ~5-10% accuracy for data privacy)\n")

    print(f"✓ Report saved: {report_path}")


def main():
    baseline_path = os.path.join(METRICS_DIR, 'baseline_history.json')
    fl_path       = os.path.join(METRICS_DIR, 'fl_history.json')

    if not os.path.exists(baseline_path):
        print("ERROR: baseline_history.json not found. Run scripts/train.py first.")
        return
    if not os.path.exists(fl_path):
        print("ERROR: fl_history.json not found. Run scripts/run_fl_server.py first.")
        return

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    with open(fl_path, 'r') as f:
        fl = json.load(f)

    print("Loaded: baseline_history.json")
    print("Loaded: fl_history.json")

    plot_baseline(baseline)
    plot_fl(fl)
    plot_comparison(baseline, fl)
    generate_report(baseline, fl)

    print("\nAll plots saved to results/plots/")
    print("Report saved to results/metrics/training_report.txt")

if __name__ == "__main__":
    main()
