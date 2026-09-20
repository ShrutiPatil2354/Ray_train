import csv
import json
import math
import os
import pickle
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .model import BikeDemandRegressor


def evaluate_checkpoint(checkpoint_dir: str, dataset: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> Dict[str, float]:
    """Load the final Ray checkpoint and evaluate it on the chronological test split."""
    with open(os.path.join(checkpoint_dir, "training_state.pkl"), "rb") as file:
        state = pickle.load(file)
    model = BikeDemandRegressor(config["input_dim"], config["hidden_dim"])
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    features = torch.tensor(dataset["test_x"], dtype=torch.float32)
    with torch.no_grad():
        scaled_predictions = model(features).numpy()
    predictions = scaled_predictions * dataset["target_scale"] + dataset["target_mean"]
    actual = np.asarray(dataset["test_y"], dtype=np.float32)
    metrics = {
        "test_mae": float(mean_absolute_error(actual, predictions)),
        "test_rmse": float(math.sqrt(mean_squared_error(actual, predictions))),
        "test_r2": float(r2_score(actual, predictions)),
        "checkpoint_epoch": int(state["epoch"]),
        "test_records": int(len(actual)),
    }
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with open(os.path.join(output_dir, "test_predictions.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "actual_cnt", "predicted_cnt"])
        for date, actual_value, predicted_value in zip(dataset["test_dates"], actual, predictions):
            writer.writerow([date, float(actual_value), float(predicted_value)])
    return metrics