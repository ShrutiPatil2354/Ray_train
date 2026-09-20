import os
import sqlite3
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.datasets import load_iris


TABLE_NAME = "iris_dataset"
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def create_database(db_path: str) -> Dict[str, Any]:
    """Load the public Iris dataset into a reproducible local SQLite database."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    dataset = load_iris()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        connection.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                id INTEGER PRIMARY KEY,
                sepal_length REAL NOT NULL,
                sepal_width REAL NOT NULL,
                petal_length REAL NOT NULL,
                petal_width REAL NOT NULL,
                target INTEGER NOT NULL,
                target_name TEXT NOT NULL
            )
            """
        )
        rows = [
            (*[float(value) for value in features], int(target), dataset.target_names[int(target)])
            for features, target in zip(dataset.data, dataset.target)
        ]
        connection.executemany(
            f"INSERT INTO {TABLE_NAME} (sepal_length, sepal_width, petal_length, petal_width, target, target_name) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return validate_database(db_path)


def validate_database(db_path: str) -> Dict[str, Any]:
    """Validate schema, row count, nulls, duplicates, and a real SQL sample query."""
    connection = sqlite3.connect(db_path)
    try:
        table_exists = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE_NAME,),
        ).fetchone()[0] == 1
        if not table_exists:
            raise ValueError(f"Required table does not exist: {TABLE_NAME}")
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
        expected_columns = ["id", *FEATURE_COLUMNS, "target", "target_name"]
        if columns != expected_columns:
            raise ValueError(f"Unexpected database schema: {columns}")
        row_count = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        null_count = connection.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE sepal_length IS NULL OR sepal_width IS NULL OR petal_length IS NULL OR petal_width IS NULL OR target IS NULL OR target_name IS NULL"
        ).fetchone()[0]
        duplicate_count = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT sepal_length, sepal_width, petal_length, petal_width, target FROM {TABLE_NAME} GROUP BY sepal_length, sepal_width, petal_length, petal_width, target HAVING COUNT(*) > 1)"
        ).fetchone()[0]
        sample_query = f"SELECT {', '.join(FEATURE_COLUMNS)}, target FROM {TABLE_NAME} ORDER BY id LIMIT 5"
        sample_rows = connection.execute(sample_query).fetchall()
    finally:
        connection.close()
    if row_count == 0 or null_count > 0:
        raise ValueError(f"Invalid database contents: rows={row_count}, nulls={null_count}")
    return {
        "database_path": os.path.abspath(db_path),
        "table": TABLE_NAME,
        "rows": row_count,
        "columns": columns,
        "null_values": null_count,
        "duplicate_groups": duplicate_count,
        "sample_query": sample_query,
        "sample_rows": [list(row) for row in sample_rows],
        "source": "scikit-learn load_iris dataset, originally Fisher (1936)",
    }


def load_dataset_from_database(db_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract features and targets using SQL; training does not read a CSV."""
    metadata = validate_database(db_path)
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            f"SELECT {', '.join(FEATURE_COLUMNS)}, target, target_name FROM {TABLE_NAME} ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    feature_count = len(FEATURE_COLUMNS)
    features = np.asarray([row[:feature_count] for row in rows], dtype=np.float32)
    targets = np.asarray([row[feature_count] for row in rows], dtype=np.int64)
    target_names = np.asarray([row[-1] for row in rows])
    return features, targets, target_names, metadata