import os
import pickle
from typing import Any, Dict

import torch
import yaml
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import SimpleRegressionNet, generate_synthetic_dataset


DEFAULT_CONFIG = {
    "seed": 42,
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_workers": 1,
    "use_gpu": True,
    "backend": "gloo",
    "dataset_size": 2000,
    "num_features": 10,
    "output_dir": "ray_train_outputs",
    "dataset_path": "data/synthetic_regression_data.csv",
}


def load_config(config_path: str = "configs/train_config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG.copy()

    with open(config_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    merged = DEFAULT_CONFIG.copy()
    merged.update(loaded)
    return merged


def build_training_plan(config_path: str = "configs/train_config.yaml") -> Dict[str, Any]:
    config = load_config(config_path)
    config["output_dir"] = os.path.abspath(config.get("output_dir", "ray_train_outputs"))
    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_directory_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, config: Dict[str, Any], output_dir: str):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
    }
    with open(os.path.join(checkpoint_dir, "training_state.pkl"), "wb") as file:
        pickle.dump(payload, file)
    return Checkpoint.from_directory(checkpoint_dir)


def _append_metrics_row(metrics_path: str, epoch: int, loss_value: float, worker_rank: int):
    file_exists = os.path.exists(metrics_path)
    with open(metrics_path, "a", encoding="utf-8") as file:
        if not file_exists:
            file.write("epoch,loss,worker\n")
        file.write(f"{epoch},{loss_value},{worker_rank}\n")


def train_loop_per_worker(config: Dict[str, Any]):
    worker_rank = train.get_context().get_world_rank()
    set_seed(int(config.get("seed", 42)))

    if torch.cuda.is_available() and bool(config.get("use_gpu", True)):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print(f"Worker {worker_rank} started")
    print(f"Worker {worker_rank} using device: {device}")
    if torch.cuda.is_available():
        print(f"Worker {worker_rank} GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print("=" * 70)

    output_dir = os.path.abspath(config.get("output_dir", "ray_train_outputs"))
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "training_metrics.csv")
    if worker_rank == 0:
        with open(metrics_path, "w", encoding="utf-8") as file:
            file.write("epoch,loss,worker\n")

    x, y = generate_synthetic_dataset(
        num_samples=int(config.get("dataset_size", 2000)),
        num_features=int(config.get("num_features", 10)),
        seed=int(config.get("seed", 42)),
    )
    model = SimpleRegressionNet(input_dim=int(config.get("num_features", 10)), hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 0.001)))
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config.get("seed", 42))),
    )

    for epoch in range(1, int(config.get("epochs", 30)) + 1):
        epoch_loss = 0.0
        sample_count = 0
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze()
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            batch_size = batch_y.shape[0]
            epoch_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        loss_value = epoch_loss / sample_count
        checkpoint = _create_directory_checkpoint(model, optimizer, epoch, config, output_dir)
        train.report(
            {
                "epoch": epoch,
                "loss": loss_value,
                "worker": worker_rank,
            },
            checkpoint=checkpoint,
        )
        _append_metrics_row(metrics_path, epoch, loss_value, worker_rank)
        print(f"Worker {worker_rank} | Epoch {epoch:02d}/{config['epochs']} | Loss: {loss_value:.6f}")

    model_path = os.path.join(output_dir, f"ray_train_model_worker_{worker_rank}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


def create_trainer(config: Dict[str, Any]):
    backend = config.get("backend", "gloo")
    if backend not in {"gloo", "nccl"}:
        backend = "gloo"

    scaling_config = ScalingConfig(num_workers=int(config.get("num_workers", 1)), use_gpu=bool(config.get("use_gpu", True)))
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config,
        torch_config=TorchConfig(backend=backend),
    )
    return trainer
