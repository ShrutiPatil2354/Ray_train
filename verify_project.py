import os
import pickle
import subprocess
import sys

import torch


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def pass_fail(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{label:<20} {status}")


def check_python():
    return sys.version_info >= (3, 10, 0)


def check_ray():
    try:
        import ray
        return ray.__version__ >= "2.58.0"
    except Exception:
        return False


def check_pytorch():
    return bool(torch.__version__)


def check_cuda():
    return torch.cuda.is_available()


def check_gpu():
    return check_cuda() and torch.cuda.device_count() > 0


def check_ray_gpu_resources():
    try:
        import ray
        ray.init(ignore_reinit_error=True)
        value = ray.available_resources().get("GPU", 0)
        ray.shutdown()
        return float(value) > 0
    except Exception:
        return False


def check_file(path):
    return os.path.exists(os.path.join(PROJECT_ROOT, path))


def check_model_and_checkpoint():
    output_dir = os.path.join(PROJECT_ROOT, "ray_train_outputs")
    if not os.path.isdir(output_dir):
        return False
    model_exists = any(name.startswith("ray_train_model_worker_") and name.endswith(".pth") for name in os.listdir(output_dir))
    checkpoint_exists = any(name.startswith("checkpoint_epoch_") for name in os.listdir(output_dir))
    return model_exists and checkpoint_exists


def check_checkpoint_restoration():
    output_dir = os.path.join(PROJECT_ROOT, "ray_train_outputs")
    if not os.path.isdir(output_dir):
        return False
    for name in os.listdir(output_dir):
        checkpoint_dir = os.path.join(output_dir, name)
        if name.startswith("checkpoint_epoch_") and os.path.isdir(checkpoint_dir):
            pickle_path = os.path.join(checkpoint_dir, "training_state.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as file:
                    payload = pickle.load(file)
                return isinstance(payload, dict) and "model_state_dict" in payload and "epoch" in payload
    return False


def check_git():
    try:
        result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def check_dvc():
    dvc_dir = os.path.join(PROJECT_ROOT, ".dvc")
    config_file = os.path.join(PROJECT_ROOT, ".dvc", "config")
    dvcignore = os.path.join(PROJECT_ROOT, ".dvcignore")
    return os.path.exists(dvc_dir) and os.path.exists(config_file) and os.path.exists(dvcignore)


print("PROJECT VERIFICATION")
print("=" * 70)
pass_fail("Python", check_python())
pass_fail("Ray", check_ray())
pass_fail("PyTorch", check_pytorch())
pass_fail("CUDA", check_cuda())
pass_fail("GPU", check_gpu())
pass_fail("Ray GPU resources", check_ray_gpu_resources())
pass_fail("Training script", check_file("ray_train_demo.py"))
pass_fail("Configuration", check_file("configs/train_config.yaml"))
pass_fail("Training metrics", check_file("ray_train_outputs/training_metrics.csv"))
pass_fail("Loss graph", check_file("ray_train_outputs/training_loss_graph.png"))
pass_fail("Model/checkpoint", check_model_and_checkpoint())
pass_fail("Checkpoint load test", check_checkpoint_restoration())
pass_fail("Git", check_git())
pass_fail("DVC", check_dvc())
pass_fail("README", check_file("README.md"))
pass_fail("Docs", os.path.isdir(os.path.join(PROJECT_ROOT, "docs")))
print("=" * 70)
