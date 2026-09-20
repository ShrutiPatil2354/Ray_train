# MLOps CCA1 Report Content

## 1. Title Page
Ray Train with PyTorch and SQLite for UCI Bike Sharing Demand Regression

## 2. Purpose of the Tool
Ray Train is the selected Model Development tool. This project uses it to execute a PyTorch regression model with GPU resource allocation, metrics reporting, and checkpointing.

## 3. Introduction to MLOps and Importance
MLOps connects data, model development, testing, version control, reproducibility, and evaluation. The database-backed workflow makes the source data, SQL extraction, preprocessing, training, checkpoint, and test metrics inspectable.

## 4. Stages of the MLOps Lifecycle
1. Data Collection: UCI Bike Sharing Dataset.
2. Data Preparation: raw `day.csv` downloaded and ingested into SQLite.
3. Data Validation: schema, row count, target nulls, duplicates, date range, and SQL sample.
4. Feature Engineering: calendar/weather variables selected; leakage columns excluded.
5. Model Development: PyTorch MLP through Ray Train.
6. Training and Evaluation: chronological validation and held-out test metrics.
7. Experiment Tracking: CSV/JSON evidence artifacts, not an external tracking server.
8. CI/CD: not selected for this focused CCA.
9. Model Serving: not selected or implemented.
10. Monitoring: not selected or implemented.
11. Governance: not selected or implemented.
12. Continuous Improvement: Git history, tests, and reproducible evidence support changes.

## 5. Tools Used in Each Stage
| CCA category | Selected tool | Why needed | Implemented/executed | Evidence |
| --- | --- | --- | --- | --- |
| Version Control and Collaboration | Git | Mandatory CCA category | Implemented; collaboration not claimed | Git history/GitHub |
| Model Development / Distributed Training | Ray Train + PyTorch | Assigned tool and parallel GPU feature | Executed with one worker/GPU | Ray logs/metrics/checkpoints |
| Structured data workflow | SQLite | Required database-backed training path | Implemented and executed | `bike_sharing.db`, SQL evidence |
| Data versioning | DVC | Existing selected project capability | Local metadata executed; no remote | `bike_sharing.db.dvc`, DVC status |
| CI/CD, serving, monitoring, governance | Not selected | Not required for this focused CCA | Not implemented | Limitations |

## 6. Working/Features of Each Tool
SQLite stores the validated Bike Sharing table and supplies SQL query results. PyTorch defines the regression network. Ray Train allocates one worker/GPU, reports train/validation losses, and wraps per-epoch state in Ray checkpoints. Git versions the project and DVC tracks local database metadata.

## 7. Advantages and Limitations
Advantages include a real public dataset, SQL-backed training input, chronological splitting, leakage prevention, GPU execution, checkpoint loading, and reproducible evidence artifacts. Limitations include one GPU, no distributed speedup, a modest dataset, no serving/monitoring/CI/CD, and no DVC remote.

## 8. Practical Implementation
`src/data_ingestion.py` downloads and validates UCI data, creates `bike_rentals`, and extracts rows through SQL. `src/preprocessing.py` fits the feature scaler on the first 70% only. `src/training.py` executes `BikeDemandRegressor` through `TorchTrainer`. `src/evaluation.py` loads epoch 40 and calculates test MAE, RMSE, and R2.

## 9. Screenshots
| Screenshot | Exact command/file | Claim proved | Report section |
| --- | --- | --- | --- |
| 1 | UCI download output or `database_evidence.json` | Real dataset source | Dataset |
| 2 | `database_evidence.json` | SQLite table, schema, row count, SQL | Database |
| 3 | `python ray_train_demo.py` | Database validation and leakage exclusion | Data validation |
| 4 | `data_split_evidence.json` | Chronological split | Preprocessing |
| 5 | Training terminal | Ray GPU resource and CUDA worker | Ray Train |
| 6 | Training terminal | Train/validation losses | Results |
| 7 | `training_loss_graph.png` | Graph from actual CSV | Results |
| 8 | `evaluation_metrics.json` | Test MAE/RMSE/R2 | Evaluation |
| 9 | `python verify_project.py` | Checkpoint load and project PASS | Verification |
| 10 | `python -m pytest -q` | Tests pass | Testing |
| 11 | Git status/log | Version control | Git |
| 12 | `python -m dvc status` | DVC metadata state | DVC |

## 10. Documentation
Technical documentation is in `README.md`, `docs/architecture.md`, `docs/execution.md`, `docs/limitations.md`, and `docs/final_verification.md`.

## 11. Real-world Use Cases
Ray Train can scale tabular demand forecasting to larger datasets and worker groups. This project executed only the single-worker, single-GPU configuration available on the development machine.

## 12. Conclusion
The project demonstrates the assigned Ray Train tool with a real UCI dataset, SQLite database retrieval, leakage-safe chronological preprocessing, GPU regression training, Ray checkpoint loading, and held-out evaluation.

## 13. Learning Outcomes
- Database-backed ML ingestion and SQL validation
- Time-aware train/validation/test splitting
- Target-leakage prevention
- Ray Train and PyTorch GPU execution
- Regression evaluation with MAE, RMSE, and R2
- Evidence-based Git, DVC, testing, and reporting

## 14. References
- UCI Bike Sharing Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Ray documentation: https://docs.ray.io/
- PyTorch documentation: https://pytorch.org/docs/
- Fanaee-T, H. and Gama, J. (2014), Event labeling combining ensemble detectors and background knowledge.