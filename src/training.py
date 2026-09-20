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

from .model import BikeDemandRegressor


DEFAULT_CONFIG = {
    "seed": 42,
    "epochs": 40,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_workers": 1,
    "use_gpu": True,
    "backend": "gloo",
    "input_dim": 11,
    "hidden_dim": 64,
    "database_path": "data/bike_sharing.db",
    "output_dir": "ray_train_outputs",
}


def load_config(config_path: str = "configs/train_config.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG.copy()
    with open(config_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    config = DEFAULT_CONFIG.copy()
    config.update(loaded)
    return config


def build_training_plan(config_path: str = "configs/train_config.yaml") -> Dict[str, Any]:
    config = load_config(config_path)
    config["output_dir"] = os.path.abspath(config.get("output_dir", "ray_train_outputs"))
    config["database_path"] = os.path.abspath(config.get("database_path", "data/bike_sharing.db"))
    return config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_directory_checkpoint(model, optimizer, epoch: int, config: Dict[str, Any], output_dir: str):
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "training_state.pkl"), "wb") as file:
        pickle.dump(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": config,
            },
            file,
        )
    return Checkpoint.from_directory(checkpoint_dir)


def _evaluate(model, features, targets, device):
    model.eval()
    with torch.no_grad():
        predictions = model(features.to(device))
        loss = nn.MSELoss()(predictions, targets.to(device)).item()
    return float(loss)


def train_loop_per_worker(train_loop_config: Dict[str, Any]):
    config = train_loop_config["config"]
    dataset = train_loop_config["dataset"]
    worker_rank = train.get_context().get_world_rank()
    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and config["use_gpu"] else "cpu")
    print(f"Worker {worker_rank} using device: {device}")
    if device.type == "cuda":
        print(f"Worker {worker_rank} GPU: {torch.cuda.get_device_name(0)}")

    train_x = torch.tensor(dataset["train_x"], dtype=torch.float32)
    train_y = torch.tensor(dataset["train_y"], dtype=torch.float32)
    validation_x = torch.tensor(dataset["validation_x"], dtype=torch.float32)
    validation_y = torch.tensor(dataset["validation_y"], dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    model = BikeDemandRegressor(config["input_dim"], config["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    loss_fn = nn.MSELoss()
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "training_metrics.csv")
    if worker_rank == 0:
        with open(metrics_path, "w", encoding="utf-8") as file:
            file.write("epoch,train_loss,val_loss,worker\n")

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_samples += len(batch_y)
        metrics = {
            "train_loss": total_loss / total_samples,
            "val_loss": _evaluate(model, validation_x, validation_y, device),
        }
        checkpoint = _create_directory_checkpoint(model, optimizer, epoch, config, output_dir)
        train.report({"epoch": epoch, "worker": worker_rank, **metrics}, checkpoint=checkpoint)
        with open(metrics_path, "a", encoding="utf-8") as file:
            file.write(f"{epoch},{metrics['train_loss']},{metrics['val_loss']},{worker_rank}\n")
        print(
            f"Worker {worker_rank} | Epoch {epoch:02d}/{config['epochs']} | "
            f"Train loss: {metrics['train_loss']:.6f} | Val loss: {metrics['val_loss']:.6f}"
        )

    torch.save(model.state_dict(), os.path.join(output_dir, f"ray_train_model_worker_{worker_rank}.pth"))


def create_trainer(config: Dict[str, Any], dataset: Dict[str, Any]):
    return TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"config": config, "dataset": dataset},
        scaling_config=ScalingConfig(num_workers=int(config["num_workers"]), use_gpu=bool(config["use_gpu"])),
        torch_config=TorchConfig(backend=config.get("backend", "gloo")),
    )


def latest_checkpoint(output_dir: str) -> str:
    checkpoints = [
        name for name in os.listdir(output_dir)
        if name.startswith("checkpoint_epoch_") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not checkpoints:
        raise FileNotFoundError("No Ray checkpoint directories were created")
    latest = max(checkpoints, key=lambda name: int(name.rsplit("_", 1)[-1]))
    return os.path.join(output_dir, latest)