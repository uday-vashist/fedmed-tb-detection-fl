"""
Entry point for FL simulation.
Runs full federated training on single machine using server.py simulation loop.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.federated.server import run_federated_simulation

def main():
    print("="*60)
    print("FEDMED FEDERATED LEARNING SIMULATION")
    print("="*60)
    print("Simulation mode: all clients on same machine")
    print("Press Ctrl+C to stop.\n")

    try:
        model, history = run_federated_simulation()
        best_round = history['val_acc'].index(max(history['val_acc'])) + 1
        print(f"\n✓ FL training complete.")
        print(f"  Best round    : {best_round}")
        print(f"  Best val acc  : {max(history['val_acc']):.2f}%")
        print(f"  Model saved   : results/models/fl_best.pth")

    except KeyboardInterrupt:
        print("\nSimulation interrupted. Partial results may be saved.")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
