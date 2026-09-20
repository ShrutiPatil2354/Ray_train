import json
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import ray
import torch

from src.data_ingestion import create_database
from src.evaluation import evaluate_checkpoint
from src.preprocessing import prepare_dataset
from src.training import build_training_plan, create_trainer, latest_checkpoint


def print_environment_summary():
    print("=" * 70)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory (GB): {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f}")
    else:
        print("GPU: N/A")
    print(f"Ray version: {ray.__version__}")
    print("=" * 70)


def save_graph_from_csv(metrics_path: str, output_dir: str):
    metrics_df = pd.read_csv(metrics_path)
    graph_path = os.path.join(output_dir, "training_loss_graph.png")
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], marker="o", label="Train loss")
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], marker="x", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error loss")
    plt.title("Ray Train - Bike Demand Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300)
    plt.close()
    return metrics_df


def main():
    config = build_training_plan()
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    tracking_uri = config["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow_experiment"])

    database_evidence = create_database(config["database_path"])
    with open(os.path.join(output_dir, "database_evidence.json"), "w", encoding="utf-8") as file:
        json.dump(database_evidence, file, indent=2)
    print(f"Database: {database_evidence['database_path']}")
    print(f"Table: {database_evidence['table']}")
    print(f"Rows: {database_evidence['rows']}")
    print(f"Columns: {len(database_evidence['columns'])}")
    print(f"Sample query: {database_evidence['sample_query']}")
    print(f"Sample result: {database_evidence['sample_rows'][0]}")
    print("Training data loaded from database: PASS")

    dataset = prepare_dataset(config["database_path"], output_dir, int(config["seed"]))
    with open(os.path.join(output_dir, "data_split_evidence.json"), "w", encoding="utf-8") as file:
        json.dump(
            {"split_sizes": dataset["split_sizes"], "split_dates": dataset["split_dates"]},
            file,
            indent=2,
        )

    with mlflow.start_run(run_name="ray-train-bike-demand") as run:
        mlflow.log_params(
            {
                "dataset": "UCI Bike Sharing day.csv",
                "database_table": database_evidence["table"],
                "dataset_rows": database_evidence["rows"],
                "target": "cnt",
                "excluded_features": "casual,registered",
                "train_records": dataset["split_sizes"]["train"],
                "validation_records": dataset["split_sizes"]["validation"],
                "test_records": dataset["split_sizes"]["test"],
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "num_workers": config["num_workers"],
                "use_gpu": config["use_gpu"],
                "backend": config["backend"],
            }
        )
        print_environment_summary()
        print("=" * 70)
        print("STARTING RAY TRAINING")
        print("=" * 70)
        ray.init(ignore_reinit_error=True)
        print(f"Ray GPU resources: {ray.available_resources().get('GPU', 0)}")
        print(f"Requested Ray workers: {config['num_workers']}")
        print(f"Requested GPU per worker: {1 if config['use_gpu'] else 0}")
        print(f"Torch backend: {config['backend']}")
        create_trainer(config, dataset).fit()

        metrics_path = os.path.join(output_dir, "training_metrics.csv")
        metrics_df = save_graph_from_csv(metrics_path, output_dir)
        for _, row in metrics_df.iterrows():
            mlflow.log_metrics(
                {"train_loss": float(row["train_loss"]), "val_loss": float(row["val_loss"])},
                step=int(row["epoch"]),
            )
        print(f"Final validation loss: {metrics_df['val_loss'].iloc[-1]:.6f}")
        checkpoint_dir = latest_checkpoint(output_dir)
        evaluation_metrics = evaluate_checkpoint(checkpoint_dir, dataset, config, output_dir)
        mlflow.log_metrics(
            {
                "test_mae": evaluation_metrics["test_mae"],
                "test_rmse": evaluation_metrics["test_rmse"],
                "test_r2": evaluation_metrics["test_r2"],
            }
        )
        mlflow.log_artifacts(output_dir, artifact_path="outputs")
        mlflow.log_artifacts(checkpoint_dir, artifact_path="final_checkpoint")
        run_evidence = {
            "tracking_uri": tracking_uri,
            "experiment": config["mlflow_experiment"],
            "run_id": run.info.run_id,
            "artifact_uri": run.info.artifact_uri,
        }
        with open(os.path.join(output_dir, "mlflow_run.json"), "w", encoding="utf-8") as file:
            json.dump(run_evidence, file, indent=2)
        mlflow.log_artifact(os.path.join(output_dir, "mlflow_run.json"), artifact_path="tracking")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Test MAE: {evaluation_metrics['test_mae']:.6f}")
        print(f"Test RMSE: {evaluation_metrics['test_rmse']:.6f}")
        print(f"Test R2: {evaluation_metrics['test_r2']:.6f}")
        print(f"Loaded checkpoint epoch: {evaluation_metrics['checkpoint_epoch']}")
        ray.shutdown()
    print("Training, evaluation, and MLflow logging completed successfully.")


if __name__ == "__main__":
    main()