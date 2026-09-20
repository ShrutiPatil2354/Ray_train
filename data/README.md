# Data Directory

This directory contains the SQLite database used by the training workflow.

`data/iris.db` is generated from the public scikit-learn Iris dataset by `src/data_ingestion.py`. Training extracts rows with SQL from the `iris_dataset` table; it does not train from the legacy synthetic CSV artifact.
