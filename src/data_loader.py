"""
Data loading for FedMed TB detection.
Folder-based labeling: data/combined/tb/ and data/combined/healthy/
Undersampling: healthy images randomly sampled to match TB count each run.
Full healthy pool used over multiple training runs.
"""

import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from src.config import *


class TBDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


def load_combined_data(data_dir=None, random_seed=None):
    """
    Load dataset with random undersampling of healthy images.

    - Full TB pool always used (~1698 images)
    - Healthy images randomly sampled to match TB count each call
    - Different sample every run if random_seed=None
    - Pass random_seed=RANDOM_SEED for reproducible val/test splits

    Returns:
        train_data, val_data, test_data as (paths, labels) tuples
    """
    if data_dir is None:
        data_dir = DATA_RAW

    print(f"Loading data from: {data_dir}")

    # Collect all images
    tb_files = (
        glob.glob(os.path.join(data_dir, "tb", "*.png")) +
        glob.glob(os.path.join(data_dir, "tb", "*.jpg")) +
        glob.glob(os.path.join(data_dir, "tb", "*.jpeg"))
    )
    healthy_files = (
        glob.glob(os.path.join(data_dir, "healthy", "*.png")) +
        glob.glob(os.path.join(data_dir, "healthy", "*.jpg")) +
        glob.glob(os.path.join(data_dir, "healthy", "*.jpeg"))
    )

    if len(tb_files) == 0 or len(healthy_files) == 0:
        raise ValueError(f"No images found in {data_dir}. Run prepare_dataset.py first.")

    print(f"  Full TB pool     : {len(tb_files)}")
    print(f"  Full healthy pool: {len(healthy_files)}")

    # Undersample healthy to match TB count
    n_tb = len(tb_files)
    random.shuffle(healthy_files)          # different sample each run
    healthy_sampled = healthy_files[:n_tb]

    print(f"  Healthy sampled  : {len(healthy_sampled)} (of {len(healthy_files)} available)")
    print(f"  Unused healthy   : {len(healthy_files) - len(healthy_sampled)} (used in future runs)")

    image_paths = tb_files + healthy_sampled
    labels      = [1] * len(tb_files) + [0] * len(healthy_sampled)

    print(f"  Total this run   : {len(image_paths)} (balanced 1:1)")

    # Stratified 70/15/15 split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=random_seed,
        stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=random_seed,
        stratify=temp_labels
    )

    print(f"\nSplit summary:")
    print(f"  Train : {len(train_paths)}")
    print(f"  Val   : {len(val_paths)}")
    print(f"  Test  : {len(test_paths)}")

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_data_loaders(data_dir=None, batch_size=BATCH_SIZE):
    train_data, val_data, test_data = load_combined_data(data_dir)

    train_loader = DataLoader(
        TBDataset(train_data[0], train_data[1], get_transforms(train=True)),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        TBDataset(val_data[0], val_data[1], get_transforms(train=False)),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        TBDataset(test_data[0], test_data[1], get_transforms(train=False)),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
