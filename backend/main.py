import io
import os
import subprocess
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


from datetime import datetime

# In-memory buffer for real-time streaming visualization
live_prediction_history = []
sniffer_process = None


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
        df = pd.DataFrame([request.features])

        if model_type in ["Ensemble", "Ensemble_Phase1"]:
            phase1_results = []
            phase1_preds = []

            for m_type in ["CNN", "LSTM", "Transformer"]:
                model, scaler, encoders = utils.load_model_and_scaler(
                    m_type, dataset_name, device
                )
                if not model:
                    continue
                X_scaled = utils.preprocess_input(
                    df.copy(), scaler, feature_cols, None, dataset_name, model_type=m_type
                )
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                with torch.no_grad():
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_class].item()

                prediction = "Attack" if pred_class == 1 else "Normal"
                phase1_preds.append(prediction)
                phase1_results.append(
                    {
                        "prediction": prediction,
                        "confidence": confidence,
                        "metric_type": "Confidence",
                    }
                )

            attacks = phase1_preds.count("Attack")
            finalPrediction = "Attack" if attacks >= 2 else "Normal"
            phase2_result = None
            zeroDayPossible = False

            if finalPrediction == "Normal" and model_type == "Ensemble":
                model, scaler, encoders = utils.load_model_and_scaler(
                    "Autoencoder", dataset_name, device
                )
                if model:
                    X_scaled = utils.preprocess_input(
                        df.copy(), scaler, feature_cols, encoders, dataset_name, model_type="Autoencoder"
                    )
                    X_tensor = torch.FloatTensor(X_scaled).to(device)
                    with torch.no_grad():
                        loss = model.reconstruction_error(X_tensor).item()
                        confidence = loss
                        pred_class = 1 if loss > 0.1 else 0
                        phase2_result = {
                            "prediction": "Attack" if pred_class == 1 else "Normal",
                            "confidence": loss,
                            "metric_type": "Reconstruction Error",
                        }
                        if pred_class == 1:
                            zeroDayPossible = True

            result = {
                "isEnsemble": True,
                "phase1": phase1_results,
                "phase2": phase2_result,
                "finalPrediction": finalPrediction,
                "zeroDayPossible": zeroDayPossible,
                "prediction": (
                    "Attack"
                    if (finalPrediction == "Attack" or zeroDayPossible)
                    else "Normal"
                ),
                "confidence": (
                    max([r["confidence"] for r in phase1_results])
                    if phase1_results
                    else 0.0
                ),
                "metric_type": model_type,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            model, scaler, encoders = utils.load_model_and_scaler(
                model_type, dataset_name, device
            )
            if not model:
                raise HTTPException(status_code=500, detail="Failed to load model")

            X_scaled = utils.preprocess_input(
                df.copy(), scaler, feature_cols, encoders, dataset_name, model_type=model_type
            )
            X_tensor = torch.FloatTensor(X_scaled).to(device)

            with torch.no_grad():
                if model_type == "Autoencoder":
                    loss = model.reconstruction_error(X_tensor).item()
                    confidence = loss
                    pred_class = 1 if loss > 0.1 else 0
                else:
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_class].item()

            result = {
                "isEnsemble": False,
                "prediction": "Attack" if pred_class == 1 else "Normal",
                "confidence": confidence,
                "metric_type": (
                    "Reconstruction Error"
                    if model_type == "Autoencoder"
                    else "Confidence"
                ),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        live_prediction_history.insert(0, result)
        if len(live_prediction_history) > 50:
            live_prediction_history.pop()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predict/history")
def get_prediction_history():
    return {"history": live_prediction_history}


class SnifferStartRequest(BaseModel):
    model: str


@app.post("/api/sniffer/start")
def start_sniffer(req: SnifferStartRequest):
    global sniffer_process
    if sniffer_process is not None and sniffer_process.poll() is None:
        sniffer_process.terminate()
        sniffer_process.wait()

    try:
        # Launch sniffer relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(project_root, "scripts", "live_sniffer.py")

        # Use current python executable or try to find one in venv
        python_exe = sys.executable
        if "venv" not in python_exe.lower() and os.path.exists(
            os.path.join(project_root, "venv", "Scripts", "python.exe")
        ):
            python_exe = os.path.join(project_root, "venv", "Scripts", "python.exe")

        cmd = [python_exe, script_path, "--model", req.model]
        sniffer_process = subprocess.Popen(cmd, cwd=project_root)

        return {"status": "started", "model": req.model, "pid": sniffer_process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start sniffer: {e}")


@app.post("/api/sniffer/stop")
def stop_sniffer():
    global sniffer_process
    if sniffer_process is not None and sniffer_process.poll() is None:
        sniffer_process.terminate()
        sniffer_process.wait()
        sniffer_process = None
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.get("/api/sniffer/status")
def sniffer_status():
    global sniffer_process
    is_running = sniffer_process is not None and sniffer_process.poll() is None
    return {"is_running": is_running}


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
        if model_type not in ["Ensemble", "Ensemble_Phase1"]:
            model, scaler, encoders = utils.load_model_and_scaler(
                model_type, dataset_name, device
            )

            if not model:
                raise HTTPException(status_code=500, detail="Failed to load model")
        else:
            _, scaler, encoders = utils.load_model_and_scaler(
                "CNN", dataset_name, device
            )

        if dataset_name == "NSL-KDD":
            missing_cols = [
                c for c in utils.NSL_KDD_COLUMNS[:-2] if c not in df.columns
            ]
            if len(missing_cols) > 20:
                df = pd.read_csv(
                    io.BytesIO(contents), header=None, names=utils.NSL_KDD_COLUMNS
                )

        X_scaled = utils.preprocess_input(
            df.copy(), scaler, feature_cols, encoders, dataset_name, model_type=model_type
        )

        batch_size = 1000
        all_preds = []
        all_probs = []

        with torch.no_grad():
            if model_type in ["Ensemble", "Ensemble_Phase1"]:
                # Load all phase 1 models
                phase1_models = {}
                for m_type in ["CNN", "LSTM", "Transformer"]:
                    m, _, _ = utils.load_model_and_scaler(m_type, dataset_name, device)
                    if m:
                        phase1_models[m_type] = m

                # Load phase 2 model if full ensemble
                ae_model = None
                if model_type == "Ensemble":
                    ae_model, _, _ = utils.load_model_and_scaler(
                        "Autoencoder", dataset_name, device
                    )

                for i in range(0, len(X_scaled), batch_size):
                    X_batch = torch.FloatTensor(X_scaled[i : i + batch_size]).to(device)

                    batch_preds = []
                    batch_probs_max = []

                    # Phase 1
                    cnn_outputs = (
                        phase1_models["CNN"](X_batch)
                        if "CNN" in phase1_models
                        else None
                    )
                    lstm_outputs = (
                        phase1_models["LSTM"](X_batch)
                        if "LSTM" in phase1_models
                        else None
                    )
                    tf_outputs = (
                        phase1_models["Transformer"](X_batch)
                        if "Transformer" in phase1_models
                        else None
                    )

                    # Iterate through the batch to vote
                    for j in range(len(X_batch)):
                        votes = 0
                        probs = []
                        if cnn_outputs is not None:
                            p = torch.softmax(cnn_outputs[j].unsqueeze(0), dim=1)
                            if torch.argmax(p, dim=1).item() == 1:
                                votes += 1
                            probs.append(p[0][1].item())
                        if lstm_outputs is not None:
                            p = torch.softmax(lstm_outputs[j].unsqueeze(0), dim=1)
                            if torch.argmax(p, dim=1).item() == 1:
                                votes += 1
                            probs.append(p[0][1].item())
                        if tf_outputs is not None:
                            p = torch.softmax(tf_outputs[j].unsqueeze(0), dim=1)
                            if torch.argmax(p, dim=1).item() == 1:
                                votes += 1
                            probs.append(p[0][1].item())

                        is_attack = votes >= 2

                        # Phase 2 (Autoencoder)
                        if not is_attack and ae_model is not None:
                            loss = ae_model.reconstruction_error(
                                X_batch[j].unsqueeze(0)
                            ).item()
                            if loss > 0.1:
                                is_attack = True
                                probs.append(loss)

                        all_preds.append(1 if is_attack else 0)
                        all_probs.append(max(probs) if probs else 0.0)
            else:
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
