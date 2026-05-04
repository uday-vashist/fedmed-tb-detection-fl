"""
Federated Learning client (simulated hospital node).
Each client trains on its local data partition and returns updated weights.
Equal IID distribution across 3 clients for now.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import flwr as fl
import numpy as np
from src.config import *
from src.model import get_model


class TBClient(fl.client.NumPyClient):
    """
    Flower FL client representing one hospital node.
    Receives global model weights → trains locally → sends back updated weights.
    """

    def __init__(self, client_id, dataset, device):
        self.client_id = client_id
        self.dataset   = dataset
        self.device    = device
        self.model     = get_model(pretrained=True, freeze_backbone=False).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        """Extract model weights as list of numpy arrays (Flower requirement)."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Load global weights into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        Receive global weights → train locally → return updated weights.

        Args:
            parameters: Global model weights from server
            config: Training config from server (epochs, lr)
        Returns:
            updated weights, number of samples, metrics dict
        """
        self.set_parameters(parameters)

        epochs = config.get("local_epochs", LOCAL_EPOCHS)
        lr     = config.get("learning_rate", LEARNING_RATE)

        loader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,   # 0 for simulation on same machine
            pin_memory=False
        )

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.model.train()

        total_loss = 0.0
        correct    = 0
        total      = 0

        for _ in range(epochs):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / (len(loader) * epochs)
        accuracy = correct / total

        print(f"  [Client {self.client_id}] Loss: {avg_loss:.4f} | Acc: {accuracy*100:.2f}%")

        return self.get_parameters(config={}), len(self.dataset), {"accuracy": accuracy}

    def evaluate(self, parameters, config):
        """
        Evaluate global model on local data.
        Called by server after aggregation each round.
        """
        self.set_parameters(parameters)
        self.model.eval()

        loader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        return avg_loss, len(self.dataset), {"accuracy": accuracy}


def create_client_datasets(full_dataset, num_clients=NUM_CLIENTS):
    """
    Split dataset equally across clients (IID).
    Each client gets ~equal number of samples.

    Args:
        full_dataset: Full TBDataset object
        num_clients: Number of simulated hospital nodes
    Returns:
        List of Subset datasets, one per client
    """
    total     = len(full_dataset)
    indices   = np.random.permutation(total)
    splits    = np.array_split(indices, num_clients)

    client_datasets = []
    for i, split_indices in enumerate(splits):
        subset = Subset(full_dataset, split_indices.tolist())
        print(f"  Client {i+1}: {len(subset)} samples")
        client_datasets.append(subset)

    return client_datasets
