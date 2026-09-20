import os
import pickle
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_ingestion import load_dataset_from_database


def prepare_dataset(db_path: str, output_dir: str, seed: int) -> Dict[str, Any]:
    """Extract from SQLite, split without leakage, and fit scaling on train only."""
    features, targets, target_names, metadata = load_dataset_from_database(db_path)
    indices = np.arange(len(targets))
    train_indices, remaining_indices = train_test_split(indices, test_size=0.4, random_state=seed, stratify=targets)
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.5,
        random_state=seed,
        stratify=targets[remaining_indices],
    )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices]).astype(np.float32)
    validation_features = scaler.transform(features[validation_indices]).astype(np.float32)
    test_features = scaler.transform(features[test_indices]).astype(np.float32)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "preprocessor.pkl"), "wb") as file:
        pickle.dump({"scaler": scaler, "feature_columns": metadata["columns"][1:5]}, file)
    return {
        "train_x": train_features,
        "train_y": targets[train_indices],
        "validation_x": validation_features,
        "validation_y": targets[validation_indices],
        "test_x": test_features,
        "test_y": targets[test_indices],
        "test_target_names": target_names[test_indices],
        "target_names": sorted(set(target_names.tolist())),
        "metadata": metadata,
        "split_sizes": {"train": len(train_indices), "validation": len(validation_indices), "test": len(test_indices)},
    }