# scripts/split_dataset.py
"""
Split combined dataset into per-client folders for FL simulation reference.
Not required for simulation (server.py handles splits in memory),
but useful for inspecting per-client data distribution.

Output structure:
    data/federated/
        client_1/healthy/  client_1/tb/
        client_2/healthy/  client_2/tb/
        client_3/healthy/  client_3/tb/
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import random
from src.config import DATA_RAW, NUM_CLIENTS, LABEL_MAP

def main():
    print("="*50)
    print("SPLITTING DATASET ACROSS CLIENTS")
    print("="*50)

    federated_dir = os.path.join(os.path.dirname(DATA_RAW), 'federated')

    for label_name in LABEL_MAP.keys():
        src_folder = os.path.join(DATA_RAW, label_name)

        if not os.path.exists(src_folder):
            print(f"WARNING: {src_folder} not found, skipping")
            continue

        files = [f for f in os.listdir(src_folder)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        splits = [files[i::NUM_CLIENTS] for i in range(NUM_CLIENTS)]

        for client_idx, client_files in enumerate(splits):
            client_folder = os.path.join(federated_dir, f'client_{client_idx+1}', label_name)
            os.makedirs(client_folder, exist_ok=True)

            for f in client_files:
                shutil.copy(
                    os.path.join(src_folder, f),
                    os.path.join(client_folder, f)
                )

            print(f"  Client {client_idx+1} | {label_name:8s}: {len(client_files)} images")

    print("\n✓ Split complete.")
    print(f"  Output: {federated_dir}")

if __name__ == "__main__":
    main()
