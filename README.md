# Ray Train CCA1: SQLite Iris Classification

## Objective
This project demonstrates the CCA Model Development tool, Ray Train, in an end-to-end workflow using a real public dataset. The Iris dataset is loaded into SQLite, extracted with SQL, validated, preprocessed, split into train/validation/test sets, and used to train a PyTorch classifier through Ray Train.

## Dataset
- Dataset: Iris dataset
- Source: `sklearn.datasets.load_iris`, originally introduced by Fisher (1936)
- Records: 150
- Features: 4 numeric measurements: sepal length, sepal width, petal length, petal width
- Target: 3 flower classes: setosa, versicolor, virginica
- Missing values: none after database validation
- Duplicate feature/target groups: checked during ingestion

## Database workflow
The training pipeline creates `data/iris.db`, with table `iris_dataset`:

| Column | Type | Purpose |
| --- | --- | --- |
| `id` | INTEGER | Primary key |
| `sepal_length`, `sepal_width` | REAL | Sepal features |
| `petal_length`, `petal_width` | REAL | Petal features |
| `target` | INTEGER | Class label |
| `target_name` | TEXT | Class name |

Training data is extracted using SQL from this table. The training worker does not read the legacy CSV artifact.

## Preprocessing
The data is split stratified into 90 training, 30 validation, and 30 test records. `StandardScaler` is fitted only on the training split and then applied to validation and test data. The fitted preprocessor is saved as `ray_train_outputs/preprocessor.pkl`.

## Model and training
The PyTorch `IrisClassifier` contains linear layers `4 -> 32 -> 16 -> 3` with ReLU activations. It uses Adam, learning rate `0.001`, cross-entropy loss, batch size `16`, and 40 epochs.

Ray configuration:
- Ray workers: 1
- GPU per worker: 1
- `use_gpu`: `true`
- Backend: Gloo
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU

The one-worker configuration is the truthful experiment for this Windows machine with one physical GPU. Ray Train supports scalable multi-worker configurations, but multi-GPU execution was not performed.

## Installation and execution
```powershell
cd D:\ray_train_cca1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python ray_train_demo.py
python make_report_artifacts.py
python verify_project.py
python -m pytest -q
python -m dvc status
```

## Generated evidence
- `data/iris.db`: SQLite database generated from the public dataset
- `ray_train_outputs/database_evidence.json`: schema, row count, and actual sample SQL result
- `ray_train_outputs/data_split_evidence.json`: split sizes and class names
- `ray_train_outputs/preprocessor.pkl`: training-fitted scaler
- `ray_train_outputs/training_metrics.csv`: train loss, validation loss, validation accuracy, worker
- `ray_train_outputs/training_loss_graph.png`: graph generated from the metrics CSV
- `ray_train_outputs/evaluation_metrics.json`: held-out test metrics
- `ray_train_outputs/test_predictions.csv`: test predictions
- `ray_train_outputs/ray_train_model_worker_0.pth`: final model state
- `ray_train_outputs/checkpoint_epoch_*`: per-epoch Ray checkpoint directories

Final measured test results from the loaded epoch-40 checkpoint:
- Test accuracy: `0.900000`
- Weighted precision: `0.902357`
- Weighted recall: `0.900000`
- Weighted F1: `0.899749`

## Ray checkpointing
Each epoch creates a directory containing model state, optimizer state, epoch, and configuration, then wraps it with `Checkpoint.from_directory(...)` and sends it through `train.report(..., checkpoint=...)`. The final checkpoint is loaded into a fresh classifier for test evaluation. A full resume-training workflow is not claimed.

## MLOps tool selection
| CCA category | Selected tool | Status |
| --- | --- | --- |
| Version Control and Collaboration | Git | Implemented and pushed; no multi-person collaboration claimed |
| Model Development / Distributed Training | Ray Train + PyTorch | Implemented and executed with one worker/GPU |
| Data storage required by workflow | SQLite | Implemented and used for SQL extraction |
| Data versioning | DVC | Local database metadata tracked; no remote configured |
| CI/CD, serving, monitoring, governance | Not selected | Not implemented because not required for this focused CCA workflow |

## Limitations
- One physical GPU was available, so actual multi-GPU or multi-node execution was not performed.
- Gloo was used for the Windows environment; NCCL was not assumed.
- The Iris dataset is small and suitable for coursework, not enterprise-scale benchmarking.
- No speedup or distributed scaling result is claimed.
- No external experiment tracker, serving API, monitoring stack, or CI service is implemented.
- No DVC remote storage is configured.

## Git and DVC
Git versions the source, configuration, tests, documentation, and DVC metadata. DVC tracks the generated SQLite database through `data/iris.db.dvc`; its local status is verified with `python -m dvc status`. The database is reproducibly regenerated by the ingestion code, and no DVC remote is claimed.

## Documentation
- [docs/architecture.md](docs/architecture.md)
- [docs/execution.md](docs/execution.md)
- [docs/limitations.md](docs/limitations.md)
- [docs/CCA1_REPORT_CONTENT.md](docs/CCA1_REPORT_CONTENT.md)