import os
import sqlite3
import urllib.request
import zipfile
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


UCI_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
TABLE_NAME = "bike_rentals"
TARGET_COLUMN = "cnt"
LEAKAGE_COLUMNS = {"casual", "registered", TARGET_COLUMN}
FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
RAW_DIR = os.path.join("data", "raw")
RAW_ZIP = os.path.join(RAW_DIR, "bike_sharing_dataset.zip")
RAW_CSV = os.path.join(RAW_DIR, "day.csv")


def download_raw_dataset() -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    if not os.path.exists(RAW_CSV):
        if not os.path.exists(RAW_ZIP):
            urllib.request.urlretrieve(UCI_URL, RAW_ZIP)
        with zipfile.ZipFile(RAW_ZIP) as archive:
            archive.extract("day.csv", RAW_DIR)
    return os.path.abspath(RAW_CSV)


def _validate_raw_frame(frame: pd.DataFrame) -> None:
    required = {"instant", "dteday", *FEATURE_COLUMNS, "casual", "registered", TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Raw bike dataset is missing columns: {sorted(missing)}")
    if frame.empty or frame[TARGET_COLUMN].isna().any() or (frame[TARGET_COLUMN] < 0).any():
        raise ValueError("Bike dataset has an invalid target column")
    if frame["dteday"].duplicated().any():
        raise ValueError("Bike dataset contains duplicate dates")


def create_database(db_path: str) -> Dict[str, Any]:
    """Download the public UCI day-level data and ingest it into SQLite."""
    raw_path = download_raw_dataset()
    frame = pd.read_csv(raw_path)
    _validate_raw_frame(frame)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        frame.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)
        connection.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE_NAME}_instant ON {TABLE_NAME}(instant)")
        connection.commit()
    finally:
        connection.close()
    return validate_database(db_path)


def validate_database(db_path: str) -> Dict[str, Any]:
    connection = sqlite3.connect(db_path)
    try:
        table_exists = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,)
        ).fetchone()[0] == 1
        if not table_exists:
            raise ValueError(f"Required table does not exist: {TABLE_NAME}")
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
        row_count = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        null_count = connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {TARGET_COLUMN} IS NULL").fetchone()[0]
        duplicate_count = connection.execute(
            f"SELECT COUNT(*) - COUNT(DISTINCT instant) FROM {TABLE_NAME}"
        ).fetchone()[0]
        sample_query = f"SELECT dteday, {', '.join(FEATURE_COLUMNS[:3])}, {TARGET_COLUMN} FROM {TABLE_NAME} ORDER BY instant LIMIT 5"
        sample_rows = connection.execute(sample_query).fetchall()
        min_date, max_date = connection.execute(f"SELECT MIN(dteday), MAX(dteday) FROM {TABLE_NAME}").fetchone()
    finally:
        connection.close()
    expected = {"instant", "dteday", *FEATURE_COLUMNS, "casual", "registered", TARGET_COLUMN}
    if not expected.issubset(columns) or row_count == 0 or null_count or duplicate_count:
        raise ValueError("Bike database validation failed")
    return {
        "database_path": os.path.abspath(db_path),
        "table": TABLE_NAME,
        "rows": row_count,
        "columns": columns,
        "target": TARGET_COLUMN,
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS - {TARGET_COLUMN}),
        "null_target_values": null_count,
        "duplicate_instant_values": duplicate_count,
        "date_range": [min_date, max_date],
        "sample_query": sample_query,
        "sample_rows": [list(row) for row in sample_rows],
        "source": UCI_URL,
    }


def load_dataset_from_database(db_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read the non-leaking feature set and target from SQLite using SQL."""
    metadata = validate_database(db_path)
    connection = sqlite3.connect(db_path)
    try:
        query = f"SELECT dteday, {', '.join(FEATURE_COLUMNS)}, {TARGET_COLUMN} FROM {TABLE_NAME} ORDER BY instant"
        frame = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    dates = frame.pop("dteday").to_numpy()
    targets = frame.pop(TARGET_COLUMN).to_numpy(dtype=np.float32)
    return frame.to_numpy(dtype=np.float32), targets, dates, metadata