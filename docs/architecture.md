# Architecture

## Actual implemented and executed architecture

```text
Public Iris dataset
        |
        v
SQLite database: data/iris.db
        |
        v
SQL extraction and validation
        |
        v
StandardScaler fitted on training split
        |
        v
Train / validation / test split
        |
        v
Ray TorchTrainer
        |
        v
Ray Worker 0 -> PyTorch IrisClassifier -> RTX 5060 GPU
        |
        v
Ray metrics and Checkpoint.from_directory
        |
        v
Evaluation metrics, predictions, model, and artifacts
```

## Scalable architecture: NOT EXECUTED on current single-GPU hardware

```text
TorchTrainer
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

Ray supports this larger configuration, but this project executed one worker and one physical GPU only.