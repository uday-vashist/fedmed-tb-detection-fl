# scripts/run_fl_client.py
"""
Distributed FL client runner.
Use this when running clients on separate machines.
For single-machine simulation, use: python scripts/run_fl_server.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import flwr as fl
from src.federated.client import TBClient
from src.data_loader import TBDataset, get_transforms, load_combined_data
from src.config import FL_SERVER_ADDRESS

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_fl_client.py <client_id>")
        print("Example: python scripts/run_fl_client.py 0")
        sys.exit(1)

    import torch
    client_id = int(sys.argv[1])
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting client {client_id}, connecting to {FL_SERVER_ADDRESS}")

    # Load this client's data partition
    train_data, _, _ = load_combined_data()
    dataset = TBDataset(train_data[0], train_data[1], transform=get_transforms(train=True))

    client = TBClient(client_id=client_id, dataset=dataset, device=device)

    fl.client.start_numpy_client(
        server_address=FL_SERVER_ADDRESS,
        client=client,
    )

if __name__ == "__main__":
    main()
