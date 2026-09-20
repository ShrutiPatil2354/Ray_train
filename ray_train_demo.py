import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import ray
import torch

from src.model import save_synthetic_dataset_csv
from src.training import build_training_plan, create_trainer


def print_environment_summary():
    print("=" * 70)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory (GB): {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f}")
    else:
        print("GPU: N/A")
    print(f"Ray version: {ray.__version__}")
    print("=" * 70)


def save_graph_from_csv(metrics_path: str, output_dir: str):
    metrics_df = pd.read_csv(metrics_path)
    graph_path = os.path.join(output_dir, "training_loss_graph.png")
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["epoch"], metrics_df["loss"], marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Ray Train - Training Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300)
    plt.close()
    print(f"Loss graph saved to: {graph_path}")
    return metrics_df


def main():
    config = build_training_plan()
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.abspath(config["dataset_path"])
    save_synthetic_dataset_csv(
        dataset_path,
        num_samples=int(config["dataset_size"]),
        num_features=int(config["num_features"]),
        seed=int(config["seed"]),
    )

    print_environment_summary()

    print("=" * 70)
    print("STARTING RAY TRAINING")
    print("=" * 70)

    ray.init(ignore_reinit_error=True)
    print(f"Ray GPU resources: {ray.available_resources().get('GPU', 0)}")
    print(f"Requested Ray workers: {config['num_workers']}")
    print(f"Requested GPU per worker: {1 if config['use_gpu'] else 0}")
    print(f"Torch backend: {config['backend']}")
    trainer = create_trainer(config)
    trainer.fit()

    metrics_path = os.path.join(output_dir, "training_metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Training metrics were not written to: {metrics_path}")

    metrics_df = save_graph_from_csv(metrics_path, output_dir)
    initial_loss = float(metrics_df["loss"].iloc[0])
    final_loss = float(metrics_df["loss"].iloc[-1])
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Absolute loss reduction: {initial_loss - final_loss:.6f}")

    checkpoint_dirs = [name for name in os.listdir(output_dir) if name.startswith("checkpoint_epoch_")]
    if checkpoint_dirs:
        print("Checkpoint directories created:")
        for name in checkpoint_dirs:
            print(os.path.join(output_dir, name))
    else:
        print("No checkpoint directories found in output_dir.")

    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print(f"Metrics CSV: {metrics_path}")
    print(f"Loss graph: {os.path.join(output_dir, 'training_loss_graph.png')}")
    print(f"Model directory: {output_dir}")

    ray.shutdown()
    print("Training completed successfully.")


if __name__ == "__main__":
    main()