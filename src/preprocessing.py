import os
import pickle
from typing import Any, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler

from .data_ingestion import load_dataset_from_database


def prepare_dataset(db_path: str, output_dir: str, seed: int) -> Dict[str, Any]:
    """Use chronological splits and fit a target/features transform on training only."""
    features, targets, dates, metadata = load_dataset_from_database(db_path)
    split_train = int(len(targets) * 0.7)
    split_validation = int(len(targets) * 0.85)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(features[:split_train]).astype(np.float32)
    validation_x = scaler.transform(features[split_train:split_validation]).astype(np.float32)
    test_x = scaler.transform(features[split_validation:]).astype(np.float32)
    target_scale = float(max(targets[:split_train].std(), 1.0))
    target_mean = float(targets[:split_train].mean())
    train_y = ((targets[:split_train] - target_mean) / target_scale).astype(np.float32)
    validation_y = ((targets[split_train:split_validation] - target_mean) / target_scale).astype(np.float32)
    test_y = targets[split_validation:].astype(np.float32)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "preprocessor.pkl"), "wb") as file:
        pickle.dump(
            {
                "scaler": scaler,
                "feature_columns": metadata["columns"],
                "target_mean": target_mean,
                "target_scale": target_scale,
                "split_strategy": "chronological 70/15/15",
            },
            file,
        )
    return {
        "train_x": train_x,
        "train_y": train_y,
        "validation_x": validation_x,
        "validation_y": validation_y,
        "test_x": test_x,
        "test_y": test_y,
        "test_dates": dates[split_validation:].tolist(),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "metadata": metadata,
        "split_sizes": {"train": len(train_x), "validation": len(validation_x), "test": len(test_x)},
        "split_dates": {
            "train_end": dates[split_train - 1],
            "validation_end": dates[split_validation - 1],
            "test_end": dates[-1],
        },
    }