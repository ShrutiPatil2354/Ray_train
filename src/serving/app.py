import os
import pickle
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_ingestion import FEATURE_COLUMNS
from src.model import BikeDemandRegressor


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(description="The 11 non-leaking Bike Sharing model features.")


def create_app(model_path: str = "ray_train_outputs/ray_train_model_worker_0.pth", preprocessor_path: str = "ray_train_outputs/preprocessor.pkl") -> FastAPI:
    with open(preprocessor_path, "rb") as file:
        preprocessing = pickle.load(file)
    model = BikeDemandRegressor()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    scaler = preprocessing["scaler"]
    target_mean = float(preprocessing["target_mean"])
    target_scale = float(preprocessing["target_scale"])

    app = FastAPI(title="Bike Demand Prediction API", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "BikeDemandRegressor"}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        received = set(request.features)
        expected = set(FEATURE_COLUMNS)
        if received != expected:
            forbidden = sorted(received.intersection({"casual", "registered", "cnt"}))
            raise HTTPException(
                status_code=422,
                detail={"message": "Exactly the 11 legitimate model features are required.", "forbidden": forbidden},
            )
        values = [[float(request.features[column]) for column in FEATURE_COLUMNS]]
        transformed = torch.tensor(scaler.transform(values), dtype=torch.float32)
        with torch.no_grad():
            scaled_prediction = float(model(transformed).item())
        prediction = scaled_prediction * target_scale + target_mean
        return {"predicted_cnt": prediction, "feature_order": FEATURE_COLUMNS}

    return app


app = create_app()