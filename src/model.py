"""
Neural network model definition for TB detection.
ResNet-50 pretrained on ImageNet, fine-tuned for binary classification (healthy=0, tb=1).
"""

import torch
import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES, DEVICE


def get_model(pretrained=True, freeze_backbone=True):
    """
    Create ResNet-50 model for TB detection.

    Args:
        pretrained: Load ImageNet weights if True.
        freeze_backbone: If True, freeze all layers except final fc.
                         Faster training, good for small datasets.
                         Set False for full fine-tuning on GPU.
    Returns:
        PyTorch model
    """
    print(f"Loading ResNet-50 (pretrained={pretrained}, freeze_backbone={freeze_backbone})...")

    if pretrained:
        model = models.resnet50(weights='IMAGENET1K_V1')
    else:
        model = models.resnet50(weights=None)

    # Freeze all backbone layers if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final FC layer: 2048 -> NUM_CLASSES (2 for binary)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    # Final layer is always trainable (new layer, requires_grad=True by default)

    print(f"✓ ResNet-50 ready | Output classes: {NUM_CLASSES} | Backbone frozen: {freeze_backbone}")
    print(f"  Total params    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


def get_model_summary(model):
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params      : {total:,}")
    print(f"Trainable params  : {trainable:,}")
    print(f"Frozen params     : {total - trainable:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    model = get_model(pretrained=True, freeze_backbone=True)
    model = model.to(DEVICE)
    get_model_summary(model)

    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    output = model(dummy_input)
    print(f"Input  shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output      : {output}")
    print("\n✓ Model test passed!")
