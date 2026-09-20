# Ray Train CCA1 Project

## 1. Project title
Ray Train with PyTorch on a Single NVIDIA GPU for MLOps CCA1

## 2. Objective
This project demonstrates a practical MLOps workflow using Ray Train and PyTorch for a synthetic regression task. The goal is to train a real neural network, monitor training loss, persist metrics and artifacts, apply Ray-native checkpointing, and document the environment and limitations honestly.

## 3. MLOps context
This assignment focuses on the MLOps lifecycle and the required tool categories. The implemented project uses:
- Version Control: Git
- Model Development: Ray Train + PyTorch
- Lightweight metric logging: metrics are reported by Ray and persisted in CSV during training
- Model Serving: not deployed in this project
- CI/CD: discussed as a future integration, not implemented in this repository
- Monitoring: not deployed in this project
- Governance: documented as a future extension

## 4. Ray Train introduction
Ray Train is used to orchestrate distributed or parallel worker execution while keeping the PyTorch training loop explicit and reproducible. The project uses a single worker configuration because only one physical GPU is available on this laptop.

## 5. Environment
- Windows 11
- Python 3.10.9
- Ray 2.58.0
- PyTorch 2.14.0+cu130
- CUDA available: True
- NVIDIA GeForce RTX 5060 Laptop GPU

## 6. Hardware
The machine contains one physical NVIDIA GPU. The configuration intentionally uses one Ray worker and the Gloo backend because NCCL is not typically available in the Windows PyTorch + CUDA environment.

## 7. Project structure
```text
ray_train_cca1/
├── README.md
├── requirements.txt
├── .gitignore
├── ray_train_demo.py
├── verify_project.py
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── training.py
│   └── utils.py
├── configs/
│   └── train_config.yaml
├── data/
│   ├── README.md
│   └── synthetic_regression_data.csv
├── docs/
│   ├── architecture.md
│   ├── execution.md
│   ├── limitations.md
│   └── CCA1_REPORT_CONTENT.md
├── ray_train_outputs/
│   ├── training_metrics.csv
│   ├── training_loss_graph.png
│   └── ray_train_model_worker_0.pth
├── tests/
│   └── test_project.py
├── .dvc/
├── .dvcignore
└── .git
```

## 8. Installation
PowerShell commands:
```powershell
cd D:\ray_train_cca1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 9. Configuration
The training configuration is loaded from `configs/train_config.yaml`.

```yaml
seed: 42
epochs: 30
batch_size: 64
learning_rate: 0.001
num_workers: 1
use_gpu: true
backend: gloo
dataset_size: 2000
num_features: 10
output_dir: ray_train_outputs
```

The project uses `TorchConfig(backend="gloo")` because this is a Windows environment and NCCL is not the default cross-platform backend for PyTorch+Ray on Windows.

## 10. How to run
```powershell
cd D:\ray_train_cca1
.\.venv\Scripts\Activate.ps1
python ray_train_demo.py
python verify_project.py
```

## 11. Expected outputs
The training script writes the following artifacts:
- `ray_train_outputs/training_metrics.csv`
- `ray_train_outputs/training_loss_graph.png`
- `ray_train_outputs/ray_train_model_worker_0.pth`
- a Ray checkpoint directory created during training

## 12. Actual experiment results
The project was executed successfully on the available hardware with:
- 1 Ray worker
- 1 physical GPU
- CUDA available: True
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- backend: gloo

Observed regression loss:
- Initial loss: 13.844763
- Final loss: 12.690871
- Absolute reduction: 1.153892

This is a real model training result, not a fabricated value.

## 13. Checkpointing
The training loop writes per-epoch directories containing serialized model state, optimizer state, epoch, and configuration data, then wraps each directory with Ray-native `Checkpoint.from_directory(...)` for `train.report(...)`. The verification script loads the serialized payload and checks its required fields; a full resume-training workflow is not claimed.

## 14. GPU verification
The script verifies:
- PyTorch version
- Ray version
- CUDA availability
- CUDA version
- GPU name
- GPU memory
- Ray GPU resources
- requested GPU resource allocation
- worker device selection

## 15. Git usage
This repository was initialized with Git.

Commands used:
```powershell
git init
git status
git log
```

Git is used for versioning code, configuration, and documentation. This project does not claim multi-developer collaboration because that did not occur in this environment.

## 16. DVC usage
DVC repository configuration was initialized when available. This project does not claim a completed remote data-versioning workflow unless the local dataset tracking file and DVC status are present.

Commands used:
```powershell
dvc init
dvc add data/synthetic_regression_data.csv
dvc status
```

## 17. Limitations
- Only one physical GPU was available.
- Actual multi-GPU execution was not performed.
- Windows environment required `gloo` rather than NCCL.
- No multi-node cluster was used.
- No production deployment was performed.
- Synthetic regression data was used for a controlled classroom exercise.

## 18. Future scalability
The implementation is structured so that it can scale to larger datasets and additional workers when suitable GPU hardware and a compatible cluster environment are available. However, this project did not execute that larger-scale configuration.

## 19. Reproducibility instructions
```powershell
cd D:\ray_train_cca1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python ray_train_demo.py
python verify_project.py
```

## 20. MLOps lifecycle overview
The assignment requires a lifecycle view. This project implements the core phases while clearly distinguishing implemented versus future capabilities.

## 21. Tools table
| MLOps Stage | Tool 1 | Tool 2 | Purpose | Implemented |
| --- | --- | --- | --- | --- |
| Version Control | Git | DVC | Code/data versioning | Yes |
| Lightweight metric logging | Ray metrics | CSV export | Track loss over epochs | Yes |
| Model Development | Ray Train | PyTorch | Distributed/parallel training | Yes |
| CI/CD | GitHub Actions | Jenkins | Automated validation | Future |
| Model Serving | FastAPI | KServe | Inference endpoint deployment | Future |
| Monitoring | Prometheus | Grafana | Runtime observability | Future |
| Governance | Model Cards | OpenMetadata | Traceability and compliance | Future |

## 22. Claim audit summary
The project claims only what was actually executed:
- Ray Train used on a single worker with GPU
- PyTorch model trained using CUDA-enabled GPU
- CSV metrics produced by the executed training loop
- loss graph generated from the recorded CSV data
- Ray checkpoint directories stored and their serialized payload validated
- Git initialized; DVC configuration initialized when available

The project does not claim multi-GPU parallelization, production deployment, or multi-node cluster execution.
