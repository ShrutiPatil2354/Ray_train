import csv
import json
import os
import pickle
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .model import IrisClassifier


def evaluate_checkpoint(checkpoint_dir: str, dataset: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> Dict[str, float]:
    """Load the final Ray checkpoint and evaluate it on the held-out test split."""
    with open(os.path.join(checkpoint_dir, "training_state.pkl"), "rb") as file:
        state = pickle.load(file)
    model = IrisClassifier(config["input_dim"], 32, config["num_classes"])
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    features = torch.tensor(dataset["test_x"], dtype=torch.float32)
    targets = np.asarray(dataset["test_y"], dtype=np.int64)
    with torch.no_grad():
        predictions = model(features).argmax(dim=1).numpy()
    metrics = {
        "test_accuracy": float(accuracy_score(targets, predictions)),
        "test_precision_weighted": float(precision_score(targets, predictions, average="weighted", zero_division=0)),
        "test_recall_weighted": float(recall_score(targets, predictions, average="weighted", zero_division=0)),
        "test_f1_weighted": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "checkpoint_epoch": int(state["epoch"]),
        "test_records": int(len(targets)),
    }
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with open(os.path.join(output_dir, "test_predictions.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["record_index", "actual", "predicted", "actual_name"])
        for index, (actual, predicted, name) in enumerate(zip(targets, predictions, dataset["test_target_names"])):
            writer.writerow([index, int(actual), int(predicted), name])
    return metrics