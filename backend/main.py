import io
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the parent directory to sys.path so 'import backend.utils' works from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend.utils as utils

app = FastAPI(title="NIDS-DL Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.get("/api/status")
def get_status():
    return {
        "status": "Online",
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "datasets": list(utils.DATASET_CONFIGS.keys()),
        "models": {
            k: list(v["model_files"].keys()) for k, v in utils.DATASET_CONFIGS.items()
        },
    }


class LivePredictionRequest(BaseModel):
    dataset_name: str
    model_type: str
    features: Dict[str, Any]


@app.post("/api/predict/live")
def predict_live(request: LivePredictionRequest):
    try:
        dataset_name = request.dataset_name
        model_type = request.model_type

        feature_cols = utils.load_feature_columns(dataset_name)
        model, scaler, encoders = utils.load_model_and_scaler(
            model_type, dataset_name, device
        )

        if not model:
            raise HTTPException(status_code=500, detail="Failed to load model")

        df = pd.DataFrame([request.features])
        X_scaled = utils.preprocess_input(df, scaler, feature_cols, None, dataset_name)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        with torch.no_grad():
            if model_type == "Autoencoder":
                loss = model.reconstruction_error(X_tensor).item()
                confidence = loss
                threshold = 0.1
                pred_class = 1 if loss > threshold else 0
            else:
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_class].item()

        return {
            "prediction": "Attack" if pred_class == 1 else "Normal",
            "confidence": confidence,
            "metric_type": (
                "Reconstruction Error" if model_type == "Autoencoder" else "Confidence"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch")
async def predict_batch(
    dataset_name: str = Form(...),
    model_type: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        feature_cols = utils.load_feature_columns(dataset_name)
        model, scaler, encoders = utils.load_model_and_scaler(
            model_type, dataset_name, device
        )

        if not model:
            raise HTTPException(status_code=500, detail="Failed to load model")

        if dataset_name == "NSL-KDD":
            missing_cols = [
                c for c in utils.NSL_KDD_COLUMNS[:-2] if c not in df.columns
            ]
            if len(missing_cols) > 20:
                df = pd.read_csv(
                    io.BytesIO(contents), header=None, names=utils.NSL_KDD_COLUMNS
                )

        X_scaled = utils.preprocess_input(
            df, scaler, feature_cols, encoders, dataset_name
        )

        batch_size = 1000
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                X_batch = torch.FloatTensor(X_scaled[i : i + batch_size]).to(device)

                if model_type == "Autoencoder":
                    losses = model.reconstruction_error(X_batch)
                    preds = (losses > 0.1).long()
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_probs.extend(losses.cpu().numpy().tolist())
                else:
                    outputs = model(X_batch)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        df_display = pd.read_csv(io.BytesIO(contents)).head(len(all_preds))
        df_display["Prediction"] = ["Attack" if p == 1 else "Normal" for p in all_preds]
        df_display["Attack_Probability"] = all_probs

        results = df_display.to_dict(orient="records")
        return {
            "total": len(all_preds),
            "attacks": sum(all_preds),
            "normal": len(all_preds) - sum(all_preds),
            "results": results[:100],  # Return top 100 for display
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
