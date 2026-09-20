import os

import torch

from src.model import SimpleRegressionNet
from src.training import build_training_plan, load_config
from verify_project import check_checkpoint_restoration, check_metrics, check_ray_train_api


def test_model_creation():
    model = SimpleRegressionNet(input_dim=10, hidden_dim=32)
    assert isinstance(model, torch.nn.Module)
    assert model(torch.randn(2, 10)).shape == (2, 1)


def test_config_loading():
    config = load_config("configs/train_config.yaml")
    assert config["epochs"] > 0
    assert config["learning_rate"] > 0
    assert config["num_workers"] >= 1
    assert config["backend"] in {"gloo", "nccl"}


def test_training_plan_has_expected_keys():
    plan = build_training_plan()
    assert "seed" in plan
    assert "epochs" in plan
    assert "output_dir" in plan
    assert "num_workers" in plan


def test_project_files_exist():
    assert os.path.exists("configs/train_config.yaml")
    assert os.path.exists("ray_train_demo.py")
    assert os.path.exists("verify_project.py")


def test_checkpoint_payload_and_metrics_are_valid():
    assert check_checkpoint_restoration()
    assert check_metrics()


def test_ray_train_api_is_available():
    assert check_ray_train_api()
