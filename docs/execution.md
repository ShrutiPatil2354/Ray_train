# Execution Evidence

## Commands
```powershell
python ray_train_demo.py
python make_report_artifacts.py
python verify_project.py
python -m pytest -q
python -m dvc status
```

## Database evidence
The final run created `data/iris.db`, table `iris_dataset`, with 150 rows and 7 columns. The executed SQL sample was:

```sql
SELECT sepal_length, sepal_width, petal_length, petal_width, target
FROM iris_dataset ORDER BY id LIMIT 5;
```

The first returned row was `[5.1, 3.5, 1.4, 0.2, 0]`. The terminal printed `Training data loaded from database: PASS`.

## Runtime evidence
- Python 3.10.9
- Ray 2.58.0
- PyTorch 2.14.0+cu130
- CUDA 13.0 available
- NVIDIA GeForce RTX 5060 Laptop GPU, approximately 7.96 GB
- Ray GPU resource: 1.0
- Requested workers: 1
- Requested GPU per worker: 1
- Backend: Gloo
- Splits: 90 train, 30 validation, 30 test

## Evaluation
The final Ray checkpoint was loaded into a fresh `IrisClassifier` and evaluated on the held-out test split. Results are stored in `ray_train_outputs/evaluation_metrics.json` and predictions in `ray_train_outputs/test_predictions.csv`.

Measured results: accuracy `0.900000`, weighted precision `0.902357`, weighted recall `0.900000`, and weighted F1 `0.899749` on 30 test records.

## Checkpoint interpretation
Checkpoint verification loads model and optimizer state, epoch, and configuration. It is state loading and inference validation, not a full resume-training test.

## Screenshot checklist
| Screenshot | Command or file | Visible evidence | Report section |
| --- | --- | --- | --- |
| 1. Environment | `python ray_train_demo.py` | Python, Ray, PyTorch, CUDA, GPU | Environment |
| 2. Database | `ray_train_outputs/database_evidence.json` | Table, rows, SQL query/result | Data pipeline |
| 3. Ray resources | Training terminal | GPU resource 1.0, worker/GPU request | Ray configuration |
| 4. Worker | Training terminal | Worker uses CUDA and RTX 5060 | Ray Train |
| 5. Training | Training terminal | Train/validation loss and accuracy | Results |
| 6. Graph | `ray_train_outputs/training_loss_graph.png` | Train and validation loss | Results |
| 7. Evaluation | `ray_train_outputs/evaluation_metrics.json` | Test accuracy, precision, recall, F1 | Evaluation |
| 8. Checkpoint | `python verify_project.py` | Ray checkpoint load PASS | Checkpointing |
| 9. Tests | `python -m pytest -q` | Test result | Testing |
| 10. Git | `git status --short --branch; git log -1 --oneline` | Version history | Version control |
| 11. DVC | `python -m dvc status` | Database tracking status | Data versioning |