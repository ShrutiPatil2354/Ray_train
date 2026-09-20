# MLOps CCA1 Report Content

## 1. Title Page
Ray Train with PyTorch and SQLite for Iris Classification

## 2. Purpose
The project demonstrates the CCA Model Development tool, Ray Train, in a complete real-data workflow. A public Iris dataset is stored in SQLite, extracted using SQL, preprocessed, split, trained with PyTorch through Ray Train, evaluated on held-out test data, and saved with Ray checkpoints.

## 3. Introduction to MLOps
MLOps applies software engineering practices to machine-learning systems so that data, code, configuration, training, evaluation, and artifacts can be reproduced and verified.

## 4. Importance of MLOps
The workflow makes the model process traceable: the database is validated, preprocessing is recorded, training metrics are saved, the checkpoint is loaded for evaluation, and Git versions the implementation.

## 5. MLOps Lifecycle
1. Data Collection: public Iris dataset loaded through scikit-learn.
2. Data Preparation: Iris records inserted into SQLite.
3. Data Validation: schema, row count, nulls, duplicates, and SQL sample checked.
4. Feature Engineering: standardization fitted only on training data.
5. Model Development: PyTorch classifier executed through Ray Train.
6. Training and Evaluation: validation metrics during training and test metrics after checkpoint loading.
7. Experiment Tracking: lightweight CSV and JSON metric artifacts, not an external tracking server.
8. CI/CD: not selected for this focused CCA workflow.
9. Model Serving: not selected or implemented.
10. Monitoring: not selected or implemented.
11. Governance: not selected or implemented.
12. Continuous Improvement: Git history, tests, and verified artifacts support future changes.

## 6. Tools Used
| CCA category | Selected tool | Why | Implemented/executed | Evidence |
| --- | --- | --- | --- | --- |
| Version Control and Collaboration | Git | Mandatory CCA category | Implemented; no multi-person collaboration claimed | Git history and GitHub repository |
| Model Development / Distributed Training | Ray Train + PyTorch | Assigned CCA tool and feature | Executed with one worker and one GPU | Training log, metrics, checkpoints |
| Database data storage | SQLite | Required for reproducible local data workflow | Implemented and executed | `data/iris.db`, SQL evidence |
| Data versioning | DVC | Existing project capability and database metadata tracking | Local tracking executed; no remote | `data/iris.db.dvc`, DVC status |
| CI/CD, serving, monitoring, governance | Not selected | Not required for this focused implementation | Not implemented | Explicit limitation |

## 7. Ray Train Introduction
Ray Train provides worker orchestration for PyTorch training. The implementation uses `TorchTrainer`, `ScalingConfig`, `TorchConfig(backend="gloo")`, `train.report`, and Ray `Checkpoint.from_directory`.

## 8. Working
The driver creates and validates SQLite, extracts records through SQL, standardizes the split data, and passes the processed arrays to the Ray worker. Worker 0 trains `IrisClassifier` on CUDA, reports train/validation metrics, and creates a checkpoint after each epoch.

## 9. Features
- Real public dataset and SQLite storage
- SQL extraction and validation
- Stratified train/validation/test split
- Training-only scaler fitting to avoid leakage
- CUDA-enabled PyTorch classification
- Ray metric reporting and per-epoch checkpointing
- Held-out test evaluation from the loaded checkpoint

## 10. Advantages
- Uses a meaningful real dataset instead of synthetic training data.
- Keeps the database local and reproducible on Windows.
- Produces evidence for every major pipeline stage.
- Uses the assigned Ray Train tool without adding unrelated infrastructure.

## 11. Limitations
- One physical GPU was available, so multi-GPU and multi-node execution was not performed.
- Gloo was used on Windows; no NCCL claim is made.
- Iris is a small academic dataset and is not an enterprise-scale benchmark.
- No speedup, cloud execution, serving, monitoring, CI/CD, or DVC remote is claimed.
- Checkpoint loading and inference were tested, but resume-training was not.

## 12. Implementation
The implementation is distributed across `src/data_ingestion.py`, `src/preprocessing.py`, `src/model.py`, `src/training.py`, and `src/evaluation.py`. The entry point is `ray_train_demo.py`.

## 13. Architecture
### Actual architecture
```text
Real Iris dataset -> SQLite -> SQL extraction -> validation -> preprocessing
-> train/validation/test -> Ray TorchTrainer -> Ray Worker 0
-> PyTorch IrisClassifier -> RTX 5060 GPU -> Ray checkpoint
-> test evaluation -> final artifacts
```

### Scalable architecture: NOT EXECUTED on current single-GPU hardware
```text
TorchTrainer -> Ray cluster -> Worker 0/GPU 0, Worker 1/GPU 1, Worker 2/GPU 2
-> distributed data-parallel training -> aggregated metrics/checkpoints
```

## 14. Configuration
The executed configuration uses 40 epochs, batch size 16, learning rate 0.001, input dimension 4, three classes, one worker, GPU enabled, Gloo backend, and `data/iris.db` as the database path.

## 15. Execution
The final workflow commands are:
```powershell
python ray_train_demo.py
python make_report_artifacts.py
python verify_project.py
python -m pytest -q
python -m dvc status
```
The run printed the database path, table, row count, SQL sample query/result, `Training data loaded from database: PASS`, CUDA worker evidence, checkpoint creation, and test evaluation.

## 16. Results
The final run used 150 database records with 90/30/30 train/validation/test splits. The held-out epoch-40 checkpoint produced test accuracy `0.900000`, weighted precision `0.902357`, weighted recall `0.900000`, and weighted F1 `0.899749` on 30 test records. Training and validation loss history is stored in `ray_train_outputs/training_metrics.csv` and graphed in `training_loss_graph.png`.

## 17. Screenshots
| Screenshot | Exact command/file | Claim proved | Report section |
| --- | --- | --- | --- |
| 1 | `python ray_train_demo.py` | Environment and GPU | Execution |
| 2 | `ray_train_outputs/database_evidence.json` | SQLite table, rows, SQL result | Data pipeline |
| 3 | Training terminal | Ray resource 1.0 and worker CUDA device | Ray Train |
| 4 | Training terminal | Train/validation metrics | Results |
| 5 | `ray_train_outputs/training_loss_graph.png` | Graph generated from CSV | Results |
| 6 | `ray_train_outputs/evaluation_metrics.json` | Test evaluation | Evaluation |
| 7 | `python verify_project.py` | Checkpoint load PASS | Checkpointing |
| 8 | `python -m pytest -q` | Tests pass | Testing |
| 9 | `git status --short --branch; git log -1 --oneline` | Git history | Version control |
| 10 | `python -m dvc status` | DVC status | Data versioning |

## 18. Real-world Use Cases
Ray Train can support larger classification workloads such as image, text, and tabular learning on multiple workers. Those scalable workloads are supported by the framework but were not executed in this project.

## 19. Conclusion
The project demonstrates the assigned Ray Train tool with a real dataset, real SQLite data access, GPU training, checkpoint loading, and held-out evaluation while remaining honest about the single-GPU limitation.

## 20. Learning Outcomes
- Database-backed ML data access and validation
- Leakage-safe preprocessing and dataset splitting
- Practical Ray Train and PyTorch execution
- Checkpoint loading and test evaluation
- Git, DVC, testing, and evidence-based reporting

## 21. References
- Ray documentation: https://docs.ray.io/
- PyTorch documentation: https://pytorch.org/docs/
- scikit-learn Iris dataset documentation: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
- Fisher, R. A. (1936), The use of multiple measurements in taxonomic problems.