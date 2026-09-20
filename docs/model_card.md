# BikeDemandRegressor Model Card

## Model
`BikeDemandRegressor`, a PyTorch multilayer perceptron trained through Ray Train.

## Purpose and Intended Use
Predict daily total bike rentals (`cnt`) for the UCI Bike Sharing day-level dataset using calendar and weather features. This is an academic local demonstration of model development, evaluation, tracking, serving, and monitoring artifacts.

## Out-of-Scope Use
This model is not presented as a production forecasting service, safety-critical system, or evidence of multi-GPU performance. It should not be used for operational decisions without additional validation.

## Dataset and Pipeline
- Dataset: UCI Bike Sharing `day.csv`
- Records: 731
- Database: SQLite table `bike_rentals`
- Target: `cnt`
- Input features: 11 calendar/weather variables
- `casual` and `registered` excluded because they leak `cnt`
- Chronological split: 511 train / 110 validation / 110 test
- Preprocessing: `StandardScaler` fitted on training data only

## Architecture and Training
- Architecture: `11 -> 64 -> 32 -> 1` with ReLU activations
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 40
- Loss: mean squared error
- Ray workers: 1
- GPU per worker: 1
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- Backend: Gloo

## Evaluation
Metrics from the loaded epoch-40 checkpoint on 110 test records:

- MAE: `901.081848`
- RMSE: `1147.685443`
- R²: `0.670831`

## Checkpoint and Tracking
Per-epoch Ray checkpoints are stored under `ray_train_outputs/checkpoint_epoch_*`. The final checkpoint is loaded for test inference. MLflow run `8187694a1403403ba30573fc2b969a9c` records parameters, train/validation loss, test metrics, and checkpoint/output artifacts.

## Limitations
- One physical GPU; multi-GPU and multi-node execution was not performed.
- The dataset is small and daily; no distributed speedup or production claim is made.
- Monitoring is a local Evidently data-drift demonstration.
- Serving is a local FastAPI demonstration.

## Reproducibility
```powershell
python ray_train_demo.py
python make_report_artifacts.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
python src.monitoring.py
```

## Version
The model card describes the verified project commit that includes the MLflow integration and the associated run artifacts.