# Final Verification Record

## Evidence status
| Area | Status | Evidence |
| --- | --- | --- |
| Environment | IMPLEMENTED + EXECUTED | `ray_train_demo.py` output: Python 3.10.9, Ray 2.58.0, PyTorch 2.14.0+cu130, CUDA 13.0, RTX 5060 |
| Dataset | IMPLEMENTED + EXECUTED | UCI Bike Sharing `day.csv`, 731 daily rows |
| Database | IMPLEMENTED + EXECUTED | SQLite `data/bike_sharing.db`, table `bike_rentals` |
| SQL retrieval | IMPLEMENTED + EXECUTED | `database_evidence.json` contains executed query and sample rows |
| Data validation | IMPLEMENTED + EXECUTED | Schema, target nulls, duplicate `instant`, date range, and leakage columns checked |
| Leakage prevention | IMPLEMENTED + EXECUTED | `casual` and `registered` excluded from `FEATURE_COLUMNS`; tests assert exclusion |
| Preprocessing | IMPLEMENTED + EXECUTED | Training-fitted `StandardScaler`, target scaling, chronological splits |
| Train/validation/test | IMPLEMENTED + EXECUTED | 511 / 110 / 110 records; dates are ordered |
| Ray Train | IMPLEMENTED + EXECUTED | `TorchTrainer`, `ScalingConfig`, `TorchConfig`, worker training function |
| GPU | IMPLEMENTED + EXECUTED | One Ray worker, one GPU resource, CUDA device, Gloo backend |
| Metrics | IMPLEMENTED + EXECUTED | 40-row `training_metrics.csv` |
| Graph | IMPLEMENTED + EXECUTED | `training_loss_graph.png` generated from the current CSV with MSE label |
| Checkpoint | IMPLEMENTED + EXECUTED | 40 Ray checkpoint directories; epoch 40 state loaded |
| Evaluation | IMPLEMENTED + EXECUTED | Test MAE, RMSE, and R2 generated from loaded checkpoint |
| Experiment tracking | IMPLEMENTED + EXECUTED | MLflow run `8187694a1403403ba30573fc2b969a9c` finished with parameters, epoch metrics, test metrics, and artifacts |
| CI/CD | IMPLEMENTED | `.github/workflows/ci.yml` defines CPU-safe install, syntax, tests, and verification steps; hosted Actions success is pending push |
| Model serving | IMPLEMENTED + EXECUTED | FastAPI `/predict` returned HTTP 200 with a real numeric prediction |
| Monitoring | IMPLEMENTED + EXECUTED | Evidently generated `ray_train_outputs/evidently_report.html` from project data/predictions |
| Governance | IMPLEMENTED | `docs/model_card.md` describes the actual model and measured run |
| Tests | IMPLEMENTED + EXECUTED | `5 passed` |
| DVC | IMPLEMENTED + EXECUTED | `python -m dvc status`: data and pipelines up to date |
| Git | IMPLEMENTED + EXECUTED | Clean branch tracking `origin/master`; project history exists |

## Final measured results
- Train/validation/test: 511 / 110 / 110
- Epochs: 40
- Final test checkpoint epoch: 40
- Test MAE: 901.081848
- Test RMSE: 1147.685443
- Test R2: 0.670831
- Final train loss: 0.073238
- Final validation loss: 0.236135

## CCA tool audit
The workspace contains no faculty PDF, so exact PDF-specific tool names could not be independently checked. This record uses the requirements pasted into the coding session.

| Category | Current project status |
| --- | --- |
| Version Control and Collaboration | Git implemented and executed; multi-developer collaboration not claimed |
| Model Development | Ray Train + PyTorch implemented and executed |
| Structured data storage | SQLite implemented and executed because the selected workflow requires a database |
| Data versioning | DVC local database metadata implemented and executed; no remote |
| Experiment Tracking | CSV/JSON evidence artifacts implemented; no external tracking service selected |
| CI/CD | Not implemented or executed |
| Model Serving | Not implemented or executed |
| Monitoring | Not implemented or executed |
| Governance | Not implemented or executed |

The last four categories must be reconciled with the official faculty PDF or selected-tool list before claiming full CCA category compliance. No unsupported tool execution is claimed.

## Exact validation commands
```powershell
python ray_train_demo.py
python make_report_artifacts.py
python verify_project.py
python -m pytest -q
python -m dvc status
git status --short --branch
git log --oneline --decorate -5
```

## Artifact consistency
- `training_metrics.csv`: exactly 40 rows, epochs 1–40
- `checkpoint_epoch_*`: exactly 40 directories, 1–40
- `evaluation_metrics.json`: `checkpoint_epoch: 40`, `test_records: 110`
- `training_loss_graph.png`: generated from `training_metrics.csv`
- `database_evidence.json`: 731-row `bike_rentals` SQLite evidence
- `data_split_evidence.json`: chronological 511/110/110 split
