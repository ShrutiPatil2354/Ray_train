# Execution Evidence

## Commands
```powershell
python ray_train_demo.py
python make_report_artifacts.py
python verify_project.py
python -m pytest -q
python -m dvc status
```

## Dataset and database
- Source: UCI Bike Sharing Dataset, archive URL in `src/data_ingestion.py`
- Database: `data/bike_sharing.db`
- Table: `bike_rentals`
- Rows: 731
- Date range: 2011-01-01 to 2012-12-31
- Target: `cnt`
- Excluded leakage columns: `casual`, `registered`
- Sample query: `SELECT dteday, season, yr, mnth, cnt FROM bike_rentals ORDER BY instant LIMIT 5`
- First sample result: `['2011-01-01', 1, 0, 1, 985]`
- Training data loaded from database: PASS

## Preprocessing
The pipeline uses an ordered chronological split: 511 training records ending 2012-05-25, 110 validation records ending 2012-09-12, and 110 test records ending 2012-12-31. Features are standardized using statistics fitted only on the training split. The target is standardized for training and inverse-transformed for reporting.

## Runtime
- Python 3.10.9
- Ray 2.58.0
- PyTorch 2.14.0+cu130
- CUDA 13.0
- NVIDIA GeForce RTX 5060 Laptop GPU, 7.96 GB
- Ray GPU resource: 1.0
- Requested workers: 1
- Requested GPU per worker: 1
- Backend: Gloo

## Results
The final run created 40 epoch checkpoints. The test evaluation loaded epoch 40 from the checkpoint and produced:
- MAE: 901.081848
- RMSE: 1147.685443
- R2: 0.670831
- Test records: 110

## MLflow tracking
The same Ray Train execution was logged to the local SQLite-backed MLflow experiment `bike-demand-ray-train`. Run ID: `8187694a1403403ba30573fc2b969a9c`. The run finished with training parameters, per-epoch train/validation losses, final MAE/RMSE/R2, output artifacts, and the final checkpoint artifact.

Artifacts are `training_metrics.csv`, `training_loss_graph.png`, `evaluation_metrics.json`, `test_predictions.csv`, `ray_train_model_worker_0.pth`, and checkpoint directories.

## Screenshot checklist
| Screenshot | Command or file | Visible evidence | Report section |
| --- | --- | --- | --- |
| 1. Dataset | UCI source URL or terminal download output | Public dataset source | Dataset |
| 2. Database | `ray_train_outputs/database_evidence.json` | Table, rows, columns, SQL sample | Database |
| 3. Validation | `python ray_train_demo.py` | Database validation and leakage exclusions | Data validation |
| 4. Preprocessing | `ray_train_outputs/data_split_evidence.json` | Chronological split dates/sizes | Preprocessing |
| 5. Ray resources | Training terminal | GPU resource 1.0 and worker request | Ray configuration |
| 6. Worker | Training terminal | CUDA and RTX 5060 | Training |
| 7. Training | Training terminal | Train/validation losses | Results |
| 8. Graph | `ray_train_outputs/training_loss_graph.png` | Loss graph generated from CSV | Results |
| 9. Evaluation | `ray_train_outputs/evaluation_metrics.json` | MAE, RMSE, R2 | Evaluation |
| 10. Checkpoint | `python verify_project.py` | Checkpoint load PASS | Checkpointing |
| 11. Tests | `python -m pytest -q` | Test result | Testing |
| 12. Git | `git status --short --branch; git log -1 --oneline` | Version history | Version control |
| 13. DVC | `python -m dvc status` | Database metadata status | Data versioning |