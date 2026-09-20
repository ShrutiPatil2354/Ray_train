# CCA1 Report Content

## 1. Title Page
Ray Train with PyTorch on a Single NVIDIA GPU for MLOps CCA1

## 2. Purpose of the Tool
This project demonstrates the practical use of Ray Train as a model-development tool within an MLOps workflow. The objective is to train a neural network using GPU-enabled PyTorch, capture metrics, persist artifacts, and document the project honestly within the hardware limits of the current machine.

## 3. Introduction to MLOps and Importance
MLOps combines machine learning development with software engineering practices such as version control, automation, reproducibility, and deployment-readiness. Ray Train is useful because it provides a distributed-training interface for PyTorch while remaining compatible with modern deep-learning workflows.

## 4. Stages of MLOps Lifecycle
1. Data Collection
2. Data Preparation
3. Data Validation
4. Feature Engineering
5. Model Development
6. Training and Evaluation
7. Experiment Tracking
8. CI/CD
9. Model Serving
10. Monitoring
11. Governance
12. Continuous Improvement

## 5. Tools Used in Each Stage
The project specifically implements the required tool categories as follows:
- Version Control: Git
- Model Development: Ray Train + PyTorch
- Lightweight metric logging: Ray `train.report` + CSV export
- Data Versioning: DVC configuration initialized; dataset tracking depends on the local DVC setup
- CI/CD: discussed as future integration
- Model Serving: not implemented in this project
- Monitoring: not implemented in this project
- Governance: documented as future best practice

## 6. Working/Features of Each Tool
- Git: used for version control and repository state capture
- Ray Train: orchestrates the training worker and `train.report` calls
- PyTorch: defines the regression model and training loop
- DVC: repository configuration was initialized, but this run does not claim a completed remote data-versioning workflow
- CSV metrics: provide reproducible training-loss history
- Ray checkpointing: wraps per-epoch directories containing pickled model, optimizer, epoch, and configuration state

## 7. Advantages and Limitations
Advantages:
- Seeded synthetic training setup with configuration-driven execution
- Metrics and artifacts produced by an executed Ray Train run
- GPU verification in code
- Concrete artifact outputs
- Honest documentation of environment constraints

Limitations:
- Single GPU only
- No multi-GPU or multi-node experiment
- Gloo backend required on Windows
- No production deployment

## 8. Practical Implementation
The project was implemented in the workspace and successfully executed using the local environment. The code performs seeded synthetic-data generation, model training, GPU detection, Ray metric reporting, CSV loss logging, loss-graph generation, and per-epoch checkpoint creation.

The verified run used Ray 2.58.0, PyTorch 2.14.0+cu130, one Ray worker, one NVIDIA GeForce RTX 5060 Laptop GPU, CUDA, and the Gloo backend. Recorded loss decreased from 13.844763 at epoch 1 to 12.690871 at epoch 30, a reduction of 1.153892. The CSV contains one clean 30-epoch run.

## 9. Screenshots
Screenshots should include:
- terminal output from environment verification
- training logs showing GPU and loss values
- output directory listing showing metrics and graph files
- project verification output
- Git status and available DVC configuration/status

## 10. Documentation
The project documentation is included in:
- README.md
- docs/architecture.md
- docs/execution.md
- docs/limitations.md

## 11. Real-world Use Cases
- large-scale deep learning with Ray clusters
- NLP and computer vision workloads
- recommendation systems
- large datasets on cloud GPU clusters

These are realistic use cases for Ray, but this current project did not execute them on a multi-node cluster.

## 12. Conclusion
The project demonstrates a real Ray Train + PyTorch workflow on an available GPU laptop. It stays honest about the limits of the environment and avoids claiming multi-GPU or production-scale results that were not executed.

## 13. Learning Outcomes
- understanding of MLOps lifecycle stages
- practical use of Ray Train for model development
- environment validation and GPU detection
- artifact generation and checkpoint-state validation
- seeded execution and documentation discipline

## 14. References
- Ray documentation: https://docs.ray.io/
- PyTorch documentation: https://pytorch.org/docs/
- MLOps lifecycle concepts from academic coursework and practical engineering experience
