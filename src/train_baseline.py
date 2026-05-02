"""
Training pipeline for centralized (non-federated) baseline model.
Two-phase training: frozen backbone → full fine-tune.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
from src.config import *
from src.model import get_model
from src.data_loader import get_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), 100 * correct / total


def train_baseline_model(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """
    Two-phase training:
      Phase 1 (first half of epochs): backbone frozen, only FC trains
      Phase 2 (second half of epochs): full fine-tune at lower LR
    """

    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)

    device = DEVICE
    phase1_epochs = num_epochs // 2
    phase2_epochs = num_epochs - phase1_epochs

    print(f"Device        : {device}")
    print(f"Total epochs  : {num_epochs} (Phase1={phase1_epochs}, Phase2={phase2_epochs})")
    print(f"Batch size    : {batch_size}")
    print(f"LR Phase 1    : {learning_rate}")
    print(f"LR Phase 2    : {learning_rate * 0.1}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Create model with frozen backbone
    print("\nCreating model...")
    model = get_model(pretrained=True, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    # ── PHASE 1: Only FC layer trains ──────────────────────────────
    print("\n" + "="*60)
    print(f"PHASE 1: Frozen backbone ({phase1_epochs} epochs)")
    print("="*60)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(phase1_epochs):
        print(f"\nEpoch {epoch+1}/{phase1_epochs} [Phase 1]")
        print("-" * 40)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'baseline_best.pth'))
            print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")

    # ── PHASE 2: Unfreeze all, fine-tune at lower LR ───────────────
    print("\n" + "="*60)
    print(f"PHASE 2: Full fine-tune ({phase2_epochs} epochs)")
    print("="*60)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate * 0.1,  # Lower LR to avoid destroying backbone weights
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(phase2_epochs):
        print(f"\nEpoch {epoch+1}/{phase2_epochs} [Phase 2]")
        print("-" * 40)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'baseline_best.pth'))
            print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")

    # Save history
    history_path = os.path.join(METRICS_DIR, 'baseline_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n✓ Training history saved to {history_path}")

    # Final test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)

    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'baseline_best.pth')))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("\n✓ TRAINING COMPLETE")
    return model, history, test_acc


if __name__ == "__main__":
    model, history, test_acc = train_baseline_model()
