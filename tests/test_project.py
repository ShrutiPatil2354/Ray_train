import os

import numpy as np
import torch

from src.data_ingestion import FEATURE_COLUMNS, TABLE_NAME, create_database, load_dataset_from_database, validate_database
from src.model import BikeDemandRegressor
from src.preprocessing import prepare_dataset
from src.training import build_training_plan, load_config
from verify_project import check_checkpoint, check_database, check_database_evidence, check_evaluation, check_metrics, check_model, check_preprocessing, check_ray_train_api


def test_database_creation_schema_and_sql(tmp_path):
    database_path = tmp_path / "bike.db"
    metadata = create_database(str(database_path))
    assert metadata["table"] == TABLE_NAME
    assert metadata["rows"] == 731
    assert set(FEATURE_COLUMNS).issubset(metadata["columns"])
    features, targets, dates, loaded_metadata = load_dataset_from_database(str(database_path))
    assert features.shape == (731, 11)
    assert targets.shape == (731,)
    assert dates[0] < dates[-1]
    assert loaded_metadata["sample_query"].startswith("SELECT")
    assert validate_database(str(database_path))["null_target_values"] == 0


def test_preprocessing_is_chronological_and_leakage_safe(tmp_path):
    database_path = tmp_path / "bike.db"
    output_dir = tmp_path / "outputs"
    create_database(str(database_path))
    dataset = prepare_dataset(str(database_path), str(output_dir), seed=42)
    assert dataset["split_sizes"] == {"train": 511, "validation": 110, "test": 110}
    assert dataset["split_dates"]["train_end"] < dataset["split_dates"]["validation_end"] < dataset["split_dates"]["test_end"]
    assert dataset["train_x"].shape == (511, 11)
    assert "casual" not in FEATURE_COLUMNS
    assert "registered" not in FEATURE_COLUMNS
    assert os.path.exists(output_dir / "preprocessor.pkl")


def test_model_creation():
    model = BikeDemandRegressor()
    assert isinstance(model, torch.nn.Module)
    assert model(torch.randn(2, 11)).shape == (2,)


def test_config_loading():
    config = load_config("configs/train_config.yaml")
    assert config["epochs"] > 0
    assert config["learning_rate"] > 0
    assert config["num_workers"] == 1
    assert config["use_gpu"] is True
    assert config["backend"] == "gloo"
    assert config["database_path"].endswith("data/bike_sharing.db")


def test_project_artifacts_and_verification_logic():
    assert check_database()
    assert check_database_evidence()
    assert check_preprocessing()
    assert check_metrics()
    assert check_model()
    assert check_checkpoint()
    assert check_evaluation()
    assert check_ray_train_api()