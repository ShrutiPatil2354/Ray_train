# Ray Train CCA1: UCI Bike Sharing Regression

## Objective
This project implements the assigned Ray Train model-development workflow using the real UCI Bike Sharing Dataset. The pipeline downloads and validates the public data, stores it in SQLite, extracts it with SQL, performs a chronological split and leakage-safe preprocessing, trains a PyTorch regression model on Ray Train, loads the checkpoint, and evaluates held-out demand predictions.

## Dataset
- Source: UCI Bike Sharing Dataset, archive URL in `src/data_ingestion.py`
- Records: 731 daily rows
- Date range: 2011-01-01 to 2012-12-31
- Target: `cnt`, total rentals
- Numeric inputs: `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`
- Excluded leakage columns: `casual`, `registered`, because `cnt = casual + registered`

## Database workflow
The active pipeline creates `data/bike_sharing.db` and table `bike_rentals`. It validates the schema, row count, target nulls, duplicate `instant` values, date range, and a sample SQL query. Training calls `load_dataset_from_database()` and reads the feature/target rows through SQL; it does not train directly from a CSV.

## Preprocessing
The ordered records are split chronologically: 511 training rows, 110 validation rows, and 110 test rows. `StandardScaler` is fitted only on the training features. The target is standardized using training statistics and inverse-transformed for evaluation. Split and database evidence are saved under `ray_train_outputs/`.

## Model
`BikeDemandRegressor` is an MLP with `11 -> 64 -> 32 -> 1` linear layers and ReLU activations. It uses Adam, learning rate `0.001`, MSE loss, batch size `32`, and 40 epochs.

## Ray configuration
- Ray 2.58.0
- `TorchTrainer`
- `ScalingConfig(num_workers=1, use_gpu=True)`
- `TorchConfig(backend="gloo")`
- One NVIDIA GeForce RTX 5060 Laptop GPU
- Multi-GPU and multi-node execution: not performed

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

If a previous local Ray process is stale, stop it with the installed Ray CLI command before rerunning the pipeline, then execute `python ray_train_demo.py` again.

## Final measured test results
The final epoch-40 checkpoint was loaded and evaluated on 110 chronological test records:
- MAE: `901.081848`
- RMSE: `1147.685443`
- R2: `0.670831`

## Artifacts
- `data/bike_sharing.db`: SQLite database
- `data/bike_sharing.db.dvc`: local DVC metadata
- `ray_train_outputs/database_evidence.json`: schema, SQL query, and sample rows
- `ray_train_outputs/data_split_evidence.json`: chronological split evidence
- `ray_train_outputs/preprocessor.pkl`: training-fitted preprocessing state
- `ray_train_outputs/training_metrics.csv`: train/validation loss history
- `ray_train_outputs/training_loss_graph.png`: graph generated from the CSV
- `ray_train_outputs/evaluation_metrics.json`: MAE, RMSE, R2
- `ray_train_outputs/test_predictions.csv`: held-out predictions
- `ray_train_outputs/ray_train_model_worker_0.pth`: final model state
- `ray_train_outputs/checkpoint_epoch_*`: Ray checkpoint directories

## Checkpointing
Each epoch writes model state, optimizer state, epoch, and configuration to a directory, wraps it with `Checkpoint.from_directory(...)`, and reports it through `train.report(..., checkpoint=...)`. The final checkpoint is loaded for test inference. Resume-training is not claimed.

## CCA tool selection
| CCA category | Selected tool | Status |
| --- | --- | --- |
| Version Control and Collaboration | Git | Implemented and pushed; no multi-person collaboration claimed |
| Model Development / Distributed Training | Ray Train + PyTorch | Implemented and executed with one worker/GPU |
| Structured data storage | SQLite | Implemented and used for SQL extraction |
| Data versioning | DVC | Local database metadata tracked; no remote configured |
| Experiment tracking, CI/CD, serving, monitoring, governance | Not selected | Not implemented because unnecessary for this focused CCA workflow |

## Limitations
- One GPU only; no multi-GPU, multi-node, speedup, or cloud claims.
- Gloo was used on Windows.
- The dataset is daily and modest in size, not an enterprise-scale benchmark.
- No external experiment tracker, serving API, monitoring stack, CI/CD service, or DVC remote is implemented.

## Documentation
- [docs/architecture.md](docs/architecture.md)
- [docs/execution.md](docs/execution.md)
- [docs/limitations.md](docs/limitations.md)
- [docs/final_verification.md](docs/final_verification.md)
- [docs/CCA1_REPORT_CONTENT.md](docs/CCA1_REPORT_CONTENT.md)