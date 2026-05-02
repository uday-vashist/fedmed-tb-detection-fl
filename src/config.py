# src/config.py
"""
Configuration file for FedMed project.
Contains all hyperparameters, paths, and settings.
"""

import os
import torch

# ==================== PATHS ====================
# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'combined')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_SPLITS = os.path.join(PROJECT_ROOT, 'data', 'splits')

# Results paths
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Create directories if they don't exist
for path in [DATA_PROCESSED, DATA_SPLITS, MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(path, exist_ok=True)

# ==================== MODEL PARAMETERS ====================
# Image preprocessing
IMG_SIZE = 224  # ResNet expects 224x224 images
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean for normalization
STD = [0.229, 0.224, 0.225]   # ImageNet std for normalization

# Model architecture
NUM_CLASSES = 2  # Binary: Normal (0) vs TB (1)
MODEL_NAME = 'resnet50'

# Training parameters
BATCH_SIZE = 16        # Adjust based on your M1 memory
NUM_EPOCHS = 10        # For Review 2, enough to see convergence
LEARNING_RATE = 0.0005  # Standard Adam learning rate
WEIGHT_DECAY = 1e-4    # L2 regularization

# Data split ratios
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.15   # 15% for validation
TEST_RATIO = 0.15  # 15% for testing

# ==================== DEVICE CONFIGURATION ====================
# Use MPS (Metal Performance Shaders) on Mac M1 if available
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# ==================== FEDERATED LEARNING PARAMETERS ====================
NUM_CLIENTS = 3              # Simulate 3 hospitals
FL_ROUNDS = 10               # Number of communication rounds
LOCAL_EPOCHS = 5             # Epochs per client per round
FL_SERVER_ADDRESS = "localhost:8080"

# ==================== RANDOM SEED ====================
RANDOM_SEED = 42  # For reproducibility

print(f"Configuration loaded. Device: {DEVICE}")
