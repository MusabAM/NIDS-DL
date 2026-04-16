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

        ENSEMBLE_MODES = ["Ensemble", "Ensemble_Phase1", "Ensemble_AE", "Ensemble_VQC"]
        if model_type in ENSEMBLE_MODES:
            phase1_results = []
            phase1_preds = []

            # Phase 1 is always CNN, LSTM, Transformer
            available_models = utils.DATASET_CONFIGS.get(dataset_name, {}).get("model_files", {}).keys()
            phase1_candidates = [m for m in ["CNN", "LSTM", "Transformer"] if m in available_models]

            for m_type in phase1_candidates:
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
                        "model": m_type,
                        "prediction": prediction,
                        "confidence": confidence,
                        "metric_type": "Confidence",
                    }
                )

            attacks = phase1_preds.count("Attack")
            # Majority vote (e.g., if 3 models, need 2+)
            vote_threshold = (len(phase1_preds) + 1) // 2
            finalPrediction = "Attack" if attacks >= vote_threshold else "Normal"
            phase2_results = []
            zeroDayPossible = False

            # Phase 2 runs only if Phase 1 says Normal
            run_ae = model_type in ["Ensemble", "Ensemble_AE"]
            run_vqc = model_type in ["Ensemble", "Ensemble_VQC"]

            if finalPrediction == "Normal":
                # Phase 2a: Autoencoder anomaly detection
                if run_ae:
                    ae_model, ae_scaler, ae_encoders = utils.load_model_and_scaler(
                        "Autoencoder", dataset_name, device
                    )
                    if ae_model:
                        X_scaled = utils.preprocess_input(
                            df.copy(), ae_scaler, feature_cols, ae_encoders, dataset_name, model_type="Autoencoder"
                        )
                        X_tensor = torch.FloatTensor(X_scaled).to(device)
                        with torch.no_grad():
                            loss = ae_model.reconstruction_error(X_tensor).item()
                            ae_threshold = utils.DATASET_CONFIGS.get(dataset_name, {}).get("autoencoder_threshold", 0.1)
                            ae_pred = 1 if loss > ae_threshold else 0
                            phase2_results.append({
                                "model": "Autoencoder",
                                "prediction": "Attack" if ae_pred == 1 else "Normal",
                                "confidence": loss,
                                "metric_type": "Reconstruction Error",
                            })
                            if ae_pred == 1:
                                zeroDayPossible = True

                # Phase 2b: VQC classification
                if run_vqc and "VQC" in available_models:
                    vqc_model, vqc_scaler, vqc_encoders = utils.load_model_and_scaler(
                        "VQC", dataset_name, device
                    )
                    if vqc_model:
                        X_scaled = utils.preprocess_input(
                            df.copy(), vqc_scaler, feature_cols, None, dataset_name, model_type="VQC"
                        )
                        X_tensor = torch.FloatTensor(X_scaled).to(device)
                        with torch.no_grad():
                            outputs = vqc_model(X_tensor)
                            vqc_probs = torch.softmax(outputs, dim=1)
                            vqc_pred = torch.argmax(vqc_probs, dim=1).item()
                            vqc_conf = vqc_probs[0][vqc_pred].item()
                            phase2_results.append({
                                "model": "VQC",
                                "prediction": "Attack" if vqc_pred == 1 else "Normal",
                                "confidence": vqc_conf,
                                "metric_type": "Confidence",
                            })
                            if vqc_pred == 1:
                                zeroDayPossible = True

            result = {
                "isEnsemble": True,
                "phase1": phase1_results,
                "phase2": phase2_results if phase2_results else None,
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
                    threshold = utils.DATASET_CONFIGS.get(dataset_name, {}).get("autoencoder_threshold", 0.1)
                    pred_class = 1 if loss > threshold else 0
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

        result["dataset_name"] = dataset_name
        result["model_type"] = model_type
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
    dataset: str = "CICIDS2018"


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

        cmd = [python_exe, script_path, "--model", req.model, "--dataset", req.dataset]
        sniffer_process = subprocess.Popen(cmd, cwd=project_root)

        return {"status": "started", "model": req.model, "dataset": req.dataset, "pid": sniffer_process.pid}
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

        ENSEMBLE_MODES = ["Ensemble", "Ensemble_Phase1", "Ensemble_AE", "Ensemble_VQC"]
        feature_cols = utils.load_feature_columns(dataset_name)
        if model_type not in ENSEMBLE_MODES:
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
            if model_type in ENSEMBLE_MODES:
                # Load Phase 1 models (always CNN, LSTM, Transformer)
                phase1_models = {}
                for m_type in ["CNN", "LSTM", "Transformer"]:
                    m, _, _ = utils.load_model_and_scaler(m_type, dataset_name, device)
                    if m:
                        phase1_models[m_type] = m

                # Load Phase 2 models based on ensemble mode
                ae_model = None
                vqc_model = None
                vqc_scaler = None
                run_ae = model_type in ["Ensemble", "Ensemble_AE"]
                run_vqc = model_type in ["Ensemble", "Ensemble_VQC"]

                if run_ae:
                    ae_model, _, _ = utils.load_model_and_scaler(
                        "Autoencoder", dataset_name, device
                    )
                if run_vqc:
                    vqc_model, vqc_scaler, _ = utils.load_model_and_scaler(
                        "VQC", dataset_name, device
                    )

                # Pre-process VQC input separately (VQC needs PCA-reduced features)
                X_scaled_vqc = None
                if vqc_model is not None and vqc_scaler is not None:
                    X_scaled_vqc = utils.preprocess_input(
                        df.copy(), vqc_scaler, feature_cols, None,
                        dataset_name, model_type="VQC"
                    )

                for i in range(0, len(X_scaled), batch_size):
                    X_batch = torch.FloatTensor(X_scaled[i : i + batch_size]).to(device)

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

                    # VQC batch outputs for Phase 2 — uses its own PCA-reduced tensor
                    vqc_outputs = None
                    if vqc_model is not None and X_scaled_vqc is not None:
                        X_vqc_batch = torch.FloatTensor(
                            X_scaled_vqc[i : i + batch_size]
                        ).to(device)
                        vqc_outputs = vqc_model(X_vqc_batch)

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

                        # Phase 2: runs only if Phase 1 says Normal
                        if not is_attack:
                            # Phase 2a: Autoencoder
                            if ae_model is not None:
                                loss = ae_model.reconstruction_error(
                                    X_batch[j].unsqueeze(0)
                                ).item()
                                ae_threshold = utils.DATASET_CONFIGS.get(dataset_name, {}).get("autoencoder_threshold", 0.1)
                                if loss > ae_threshold:
                                    is_attack = True
                                    probs.append(loss)

                            # Phase 2b: VQC (uses its own PCA-reduced output)
                            if vqc_outputs is not None:
                                vp = torch.softmax(vqc_outputs[j].unsqueeze(0), dim=1)
                                if torch.argmax(vp, dim=1).item() == 1:
                                    is_attack = True
                                probs.append(vp[0][1].item())

                        all_preds.append(1 if is_attack else 0)
                        all_probs.append(max(probs) if probs else 0.0)
            else:
                for i in range(0, len(X_scaled), batch_size):
                    X_batch = torch.FloatTensor(X_scaled[i : i + batch_size]).to(device)

                    if model_type == "Autoencoder":
                        losses = model.reconstruction_error(X_batch)
                        threshold = utils.DATASET_CONFIGS.get(dataset_name, {}).get("autoencoder_threshold", 0.1)
                        preds = (losses > threshold).long()
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
