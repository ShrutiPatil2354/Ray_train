import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data_ingestion import FEATURE_COLUMNS, load_dataset_from_database


def generate_evidently_report(database_path: str, predictions_path: str, output_path: str) -> str:
    features, targets, _, _ = load_dataset_from_database(database_path)
    predictions = pd.read_csv(predictions_path)
    reference = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    reference["cnt"] = targets
    current = reference.iloc[-len(predictions):].copy()
    current["prediction"] = predictions["predicted_cnt"].to_numpy()
    reference["prediction"] = reference["cnt"]
    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result.save_html(output_path)
    return output_path


if __name__ == "__main__":
    from src.training import build_training_plan

    config = build_training_plan()
    path = generate_evidently_report(
        config["database_path"],
        "ray_train_outputs/test_predictions.csv",
        "ray_train_outputs/evidently_report.html",
    )
    print(f"Evidently report: {path}")