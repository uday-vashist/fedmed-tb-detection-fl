# scripts/train.py
"""
Entry point for baseline (centralized) training.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_baseline import train_baseline_model

def main():
    print("="*70)
    print(" "*20 + "FEDMED BASELINE TRAINING")
    print("="*70)
    print("\nCentralized ResNet-50 training for TB detection.")
    print("Expected time: 10-20 min on GPU, 30-40 min on M1\n")

    try:
        model, history, test_acc = train_baseline_model()
        print("\n" + "="*70)
        print(f"TRAINING COMPLETE | Test Accuracy: {test_acc:.2f}%")
        print("="*70)
        print("\nNext: python scripts/plot_results.py")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Partial results saved in results/")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
