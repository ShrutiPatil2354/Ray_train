# Architecture

## Actual implemented architecture

```text
UCI Bike Sharing Dataset
        |
        v
Raw day.csv acquisition and validation
        |
        v
SQLite data/bike_sharing.db
        |
        v
SQL query from bike_rentals
        |
        v
Validation and leakage-column removal
        |
        v
Chronological train / validation / test split
        |
        v
StandardScaler fitted on training data only
        |
        v
Ray TorchTrainer
        |
        v
Ray Worker 0 -> PyTorch BikeDemandRegressor -> RTX 5060 GPU
        |
        v
Ray metrics and Checkpoint.from_directory
        |
        v
Loaded checkpoint -> test MAE/RMSE/R2 -> predictions/model artifacts
```

The model excludes `casual` and `registered` because `cnt = casual + registered` and using them would leak the target.

## Scalable architecture: NOT EXECUTED on current single-GPU hardware

```text
Ray Train
    |
    v
Ray cluster
    |
    +--> Worker 0 -> GPU 0
    +--> Worker 1 -> GPU 1
    +--> Worker 2 -> GPU 2
    |
    v
Distributed data-parallel training
```

Ray supports this larger configuration, but the actual experiment used one worker and one physical GPU only.