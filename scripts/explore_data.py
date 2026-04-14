# scripts/explore_data.py
"""
Visualize dataset samples and distribution.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from PIL import Image
from src.data_loader import load_combined_data
from src.config import PLOTS_DIR, LABEL_MAP

def main():
    print("="*50)
    print("EXPLORING FEDMED DATASET")
    print("="*50)

    train_data, val_data, test_data = load_combined_data()
    train_paths, train_labels = train_data

    total = len(train_paths) + len(val_data[0]) + len(test_data[0])
    print(f"\nTotal images : {total}")
    print(f"Train        : {len(train_paths)}")
    print(f"  Healthy (0): {train_labels.count(0)}")
    print(f"  TB      (1): {train_labels.count(1)}")
    print(f"Val          : {len(val_data[0])}")
    print(f"Test         : {len(test_data[0])}")

    # Sample 3 healthy + 3 TB
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sample TB X-ray Images', fontsize=16, fontweight='bold')

    for row, (label_val, label_name, color) in enumerate([(0, 'Healthy', 'green'), (1, 'TB', 'red')]):
        indices = [i for i, l in enumerate(train_labels) if l == label_val]
        collected = []
        for idx in indices:
            try:
                img = Image.open(train_paths[idx])
                collected.append(img)
                if len(collected) == 3:
                    break
            except:
                continue
        for col, img in enumerate(collected):
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(label_name, fontsize=14, color=color)
            axes[row, col].axis('off')

    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'data_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
