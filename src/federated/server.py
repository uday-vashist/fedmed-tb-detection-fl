"""
Federated Learning server.
Coordinates FL rounds, aggregates client weights using FedAvg.
Simulation mode: all clients run on same machine.
"""

import flwr as fl
from torch.utils.data import DataLoader
from flwr.server.strategy import FedAvg
import torch.nn as nn
from torchvision import transforms
import torch
import numpy as np
from src.config import *
from src.model import get_model
from src.data_loader import TBDataset, get_transforms, load_combined_data
from src.federated.client import TBClient, create_client_datasets


def get_initial_weights():
    """Get initial global model weights to send to clients on round 1."""
    model = get_model(pretrained=False, freeze_backbone=False)
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def run_federated_simulation():
    """
    Run full FL simulation on single machine.

    Flow:
      1. Load full dataset, split across NUM_CLIENTS
      2. Initialize global model
      3. For each FL round:
           a. Send global weights to all clients
           b. Each client trains locally
           c. Server aggregates weights (FedAvg)
           d. Evaluate global model
    """

    print("\n" + "="*60)
    print("FEDERATED LEARNING SIMULATION")
    print(f"Clients : {NUM_CLIENTS} simulated hospitals")
    print(f"Rounds  : {FL_ROUNDS}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print("="*60)

    device = DEVICE

    # Load full training data
    print("\nLoading and splitting data across clients...")
    train_data, val_data, _ = load_combined_data()

    train_dataset = TBDataset(
        train_data[0], train_data[1],
        transform=get_transforms(train=True)
    )
    val_dataset = TBDataset(
        val_data[0], val_data[1],
        transform=get_transforms(train=False)
    )

    # Split training data across clients
    print("Client data distribution:")
    client_datasets = create_client_datasets(train_dataset, NUM_CLIENTS)

    # Create client instances
    clients = [
        TBClient(client_id=i, dataset=client_datasets[i], device=device)
        for i in range(NUM_CLIENTS)
    ]

    # Global model weights (initial)
    global_weights = get_initial_weights()

    # Training history
    history = {
        'round': [],
        'avg_train_acc': [],
        'val_acc': [],
        'val_loss': []
    }

    print("\n" + "="*60)
    print("STARTING FL ROUNDS")
    print("="*60)

    for fl_round in range(1, FL_ROUNDS + 1):
        print(f"\n── Round {fl_round}/{FL_ROUNDS} ──")

        # ── Client training ────────────────────────────────────────
        all_weights  = []
        all_sizes    = []
        round_accs   = []

        config = {"local_epochs": LOCAL_EPOCHS, "learning_rate": LEARNING_RATE}

        for client in clients:
            weights, size, metrics = client.fit(global_weights, config)
            all_weights.append(weights)
            all_sizes.append(size)
            round_accs.append(metrics["accuracy"])

      # ── FedAvg aggregation ─────────────────────────────────────
        total_samples = sum(all_sizes)
        global_weights = [
            np.sum(
                [all_weights[c][layer] * (all_sizes[c] / total_samples)
                 for c in range(NUM_CLIENTS)],
                axis=0
            )
            for layer in range(len(all_weights[0]))
        ]

        avg_train_acc = np.mean(round_accs) * 100

        # ── Build global model from aggregated weights ──────────────
        global_model = get_model(pretrained=False, freeze_backbone=False).to(device)
        params_dict  = zip(global_model.state_dict().keys(), global_weights)
        state_dict   = {k: torch.tensor(v) for k, v in params_dict}
        global_model.load_state_dict(state_dict, strict=True)

        # ── Centralized evaluation on held-out val set ──────────────
        # val_dataset is the actual validation split, not client training data
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        criterion = nn.CrossEntropyLoss()
        global_model.eval()

        val_loss_total = 0.0
        val_correct    = 0
        val_total      = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = global_model(images)
                loss    = criterion(outputs, labels)
                val_loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total   += labels.size(0)

        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_acc  = (val_correct / val_total) * 100

        print(f"  Avg Train Acc : {avg_train_acc:.2f}%")
        print(f"  Avg Val   Acc : {avg_val_acc:.2f}%")
        print(f"  Avg Val   Loss: {avg_val_loss:.4f}")

        history['round'].append(fl_round)
        history['avg_train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['val_loss'].append(avg_val_loss)

        # ── Save global model checkpoint ────────────────────────────
        torch.save(
            global_model.state_dict(),
            f"{MODELS_DIR}/fl_global_round_{fl_round}.pth"
        )
    # Save best round model separately
    best_round = np.argmax(history['val_acc'])
    print(f"\n✓ Best round: {best_round+1} (Val Acc: {history['val_acc'][best_round]:.2f}%)")

    import shutil
    shutil.copy(
        f"{MODELS_DIR}/fl_global_round_{best_round+1}.pth",
        f"{MODELS_DIR}/fl_best.pth"
    )
    print(f"✓ Best model saved as fl_best.pth")

    import json, os
    history_path = os.path.join(METRICS_DIR, 'fl_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"✓ FL history saved to {history_path}")

    return global_model, history


if __name__ == "__main__":
    model, history = run_federated_simulation()
