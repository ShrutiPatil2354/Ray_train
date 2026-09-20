# Architecture Documentation

## Actual executed architecture
The project executed the following path on the current laptop:

User configuration
        ↓
Ray Train + TorchTrainer
        ↓
Single Ray worker
        ↓
PyTorch neural network
        ↓
NVIDIA RTX 5060 Laptop GPU
        ↓
Ray `train.report` metrics
        ↓
Ray `Checkpoint.from_directory`
        ↓
CSV, graph, model, and checkpoint artifacts

## Important note
This is an executed architecture for a single-worker, single-GPU laptop setup. It is not a multi-node cluster architecture.

## Scalable configuration — NOT EXECUTED on the current hardware
Ray Train supports a scalable configuration such as:

User configuration
        ↓
Ray cluster scheduler
        ↓
Multiple Ray workers
        ↓
Distributed data-parallel workers
        ↓
PyTorch model replicas
        ↓
Multiple GPUs or nodes
        ↓
Aggregated metrics and checkpointing

This configuration is supported by the framework, but was not experimentally demonstrated because only one physical GPU was available.

## Components in this project
- Ray scheduler: active during training
- TorchTrainer: active during training
- Model: executed in Python
- GPU: one physical NVIDIA GPU detected and used
- Metrics: stored in CSV
- Checkpoint: stored using Ray Checkpoint
- Artifacts: saved to `ray_train_outputs`
