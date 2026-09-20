import csv
import json
import os
import pickle
import subprocess
import sys

import matplotlib.image as mpimg
import torch

from src.data_ingestion import FEATURE_COLUMNS, LEAKAGE_COLUMNS, TABLE_NAME, validate_database
from src.model import BikeDemandRegressor
from src.training import build_training_plan


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def pass_fail(label: str, condition: bool) -> None:
    print(f"{label:<28} {'PASS' if condition else 'FAIL'}")


def check_python():
    return sys.version_info >= (3, 10)


def check_ray():
    try:
        import ray
        return ray.__version__ >= "2.58.0"
    except Exception:
        return False


def check_ray_train_api():
    try:
        from ray.train import Checkpoint, ScalingConfig
        from ray.train.torch import TorchConfig, TorchTrainer
        from src.training import train_loop_per_worker
        return all((Checkpoint, ScalingConfig, TorchConfig, TorchTrainer, train_loop_per_worker))
    except Exception:
        return False


def check_ray_gpu_resources():
    try:
        import ray
        ray.init(ignore_reinit_error=True)
        value = float(ray.available_resources().get("GPU", 0))
        ray.shutdown()
        return value >= 1
    except Exception:
        return False


def check_file(path):
    return os.path.exists(os.path.join(PROJECT_ROOT, path))


def check_configuration():
    config = build_training_plan()
    return config["input_dim"] == len(FEATURE_COLUMNS) and config["num_workers"] == 1 and config["use_gpu"] is True and config["backend"] == "gloo"


def check_database():
    try:
        metadata = validate_database(build_training_plan()["database_path"])
        return metadata["table"] == TABLE_NAME and metadata["rows"] == 731 and metadata["target"] == "cnt" and set(metadata["excluded_leakage_columns"]) == LEAKAGE_COLUMNS - {"cnt"}
    except Exception:
        return False


def check_database_evidence():
    path = os.path.join(PROJECT_ROOT, "ray_train_outputs", "database_evidence.json")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as file:
        evidence = json.load(file)
    return evidence.get("table") == TABLE_NAME and evidence.get("rows") == 731 and evidence.get("sample_query", "").startswith("SELECT")


def check_preprocessing():
    path = os.path.join(PROJECT_ROOT, "ray_train_outputs", "data_split_evidence.json")
    preprocessor = os.path.join(PROJECT_ROOT, "ray_train_outputs", "preprocessor.pkl")
    if not os.path.isfile(path) or not os.path.isfile(preprocessor):
        return False
    with open(path, encoding="utf-8") as file:
        evidence = json.load(file)
    return evidence.get("split_sizes") == {"train": 511, "validation": 110, "test": 110}


def check_metrics():
    path = os.path.join(PROJECT_ROOT, "ray_train_outputs", "training_metrics.csv")
    if not os.path.isfile(path):
        return False
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or set(rows[0]) != {"epoch", "train_loss", "val_loss", "worker"}:
        return False
    try:
        epochs = [int(row["epoch"]) for row in rows]
        values = [[float(row["train_loss"]), float(row["val_loss"])] for row in rows]
        workers = [int(row["worker"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return epochs == list(range(1, len(rows) + 1)) and all(torch.isfinite(torch.tensor(row)).all().item() for row in values) and workers == [0] * len(rows)


def check_graph():
    try:
        image = mpimg.imread(os.path.join(PROJECT_ROOT, "ray_train_outputs", "training_loss_graph.png"))
        return image.ndim in {2, 3} and image.shape[0] > 0 and image.shape[1] > 0
    except Exception:
        return False


def _load_checkpoint_state():
    output_dir = os.path.join(PROJECT_ROOT, "ray_train_outputs")
    names = [name for name in os.listdir(output_dir) if name.startswith("checkpoint_epoch_")]
    latest = max(names, key=lambda name: int(name.rsplit("_", 1)[-1]))
    with open(os.path.join(output_dir, latest, "training_state.pkl"), "rb") as file:
        return pickle.load(file)


def check_checkpoint():
    try:
        state = _load_checkpoint_state()
        config = state["config"]
        model = BikeDemandRegressor(config["input_dim"], config["hidden_dim"])
        model.load_state_dict(state["model_state_dict"])
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        optimizer.load_state_dict(state["optimizer_state_dict"])
        return state["epoch"] == config["epochs"]
    except Exception:
        return False


def check_model():
    try:
        state = torch.load(os.path.join(PROJECT_ROOT, "ray_train_outputs", "ray_train_model_worker_0.pth"), map_location="cpu", weights_only=True)
        model = BikeDemandRegressor()
        model.load_state_dict(state)
        return True
    except Exception:
        return False


def check_evaluation():
    path = os.path.join(PROJECT_ROOT, "ray_train_outputs", "evaluation_metrics.json")
    predictions = os.path.join(PROJECT_ROOT, "ray_train_outputs", "test_predictions.csv")
    if not os.path.isfile(path) or not os.path.isfile(predictions):
        return False
    with open(path, encoding="utf-8") as file:
        metrics = json.load(file)
    return metrics.get("test_records") == 110 and metrics.get("checkpoint_epoch") == 40 and all(key in metrics for key in ("test_mae", "test_rmse", "test_r2"))


def check_git():
    result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def check_git_history():
    result = subprocess.run(["git", "log", "-1", "--oneline"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


def check_dvc():
    return all(check_file(path) for path in (".dvc", ".dvc/config", ".dvcignore", "data/bike_sharing.db.dvc"))


def main():
    print("PROJECT VERIFICATION")
    print("=" * 82)
    pass_fail("Python", check_python())
    pass_fail("Ray", check_ray())
    pass_fail("Ray Train API", check_ray_train_api())
    pass_fail("PyTorch", bool(torch.__version__))
    pass_fail("CUDA", torch.cuda.is_available())
    pass_fail("Ray GPU resources", check_ray_gpu_resources())
    pass_fail("Configuration", check_configuration())
    pass_fail("Bike database/schema", check_database())
    pass_fail("Database evidence", check_database_evidence())
    pass_fail("Preprocessing/splits", check_preprocessing())
    pass_fail("Training metrics", check_metrics())
    pass_fail("Loss graph", check_graph())
    pass_fail("Model artifact", check_model())
    pass_fail("Ray checkpoint load", check_checkpoint())
    pass_fail("Test evaluation", check_evaluation())
    pass_fail("Git", check_git())
    pass_fail("Git history", check_git_history())
    pass_fail("DVC database metadata", check_dvc())
    pass_fail("Documentation", check_file("README.md") and os.path.isdir(os.path.join(PROJECT_ROOT, "docs")))
    print("=" * 82)


if __name__ == "__main__":
    main()