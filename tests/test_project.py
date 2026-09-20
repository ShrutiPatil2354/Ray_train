import os

import torch

from src.data_ingestion import FEATURE_COLUMNS, TABLE_NAME, create_database, load_dataset_from_database, validate_database
from src.model import IrisClassifier
from src.preprocessing import prepare_dataset
from src.training import build_training_plan, load_config
from verify_project import (
    check_checkpoint,
    check_database,
    check_database_evidence,
    check_evaluation,
    check_metrics,
    check_model,
    check_preprocessing,
    check_ray_train_api,
)


def test_database_creation_schema_and_sql(tmp_path):
    database_path = tmp_path / "iris.db"
    metadata = create_database(str(database_path))
    assert metadata["table"] == TABLE_NAME
    assert metadata["rows"] == 150
    assert metadata["columns"] == ["id", *FEATURE_COLUMNS, "target", "target_name"]
    features, targets, names, loaded_metadata = load_dataset_from_database(str(database_path))
    assert features.shape == (150, 4)
    assert targets.shape == (150,)
    assert names.shape == (150,)
    assert loaded_metadata["sample_query"].startswith("SELECT")
    assert validate_database(str(database_path))["null_values"] == 0


def test_preprocessing_split_and_scaler(tmp_path):
    database_path = tmp_path / "iris.db"
    output_dir = tmp_path / "outputs"
    create_database(str(database_path))
    dataset = prepare_dataset(str(database_path), str(output_dir), seed=42)
    assert dataset["split_sizes"] == {"train": 90, "validation": 30, "test": 30}
    assert dataset["train_x"].shape == (90, 4)
    assert os.path.exists(output_dir / "preprocessor.pkl")


def test_model_creation():
    model = IrisClassifier()
    assert isinstance(model, torch.nn.Module)
    assert model(torch.randn(2, 4)).shape == (2, 3)


def test_config_loading():
    config = load_config("configs/train_config.yaml")
    assert config["epochs"] > 0
    assert config["learning_rate"] > 0
    assert config["num_workers"] == 1
    assert config["use_gpu"] is True
    assert config["backend"] == "gloo"
    assert config["database_path"].endswith("data/iris.db")


def test_training_plan_has_expected_keys():
    plan = build_training_plan()
    assert plan["input_dim"] == 4
    assert plan["num_classes"] == 3
    assert os.path.isabs(plan["database_path"])


def test_project_artifacts_and_verification_logic():
    assert check_database()
    assert check_database_evidence()
    assert check_preprocessing()
    assert check_metrics()
    assert check_model()
    assert check_checkpoint()
    assert check_evaluation()
    assert check_ray_train_api()