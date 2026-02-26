"""
Train an LSTM classifier on the CICIDS2017 dataset.
Uses the improved LSTM model from src/models/classical/lstm.py.
"""

import os
import sys
import glob
import pickle
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical.lstm import create_lstm_torch

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cicids2017")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "cicids2017_scaler.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "cicids2017_feature_cols.pkl")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_lstm_cicids2017.pth")
RESULTS_PATH = os.path.join(RESULTS_DIR, "lstm_cicids2017_results.txt")

BATCH_SIZE = 1024
EPOCHS = 10  # Reduced for speed, can be increased
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_binary_label(label):
    if isinstance(label, str) and "BENIGN" in label.upper():
        return 0
    return 1

def load_data():
    print("Loading data...")
    all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    li = []
    for filename in all_files:
        print(f"  Reading {os.path.basename(filename)}...")
        df_temp = pd.read_csv(filename, encoding='cp1252', low_memory=True)
        li.append(df_temp)
    
    df = pd.concat(li, axis=0, ignore_index=True)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df["binary_label"] = df["Label"].apply(create_binary_label)
    
    # Load feature columns if available, else use all
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, "rb") as f:
            feature_cols = pickle.load(f)
    else:
        # Same logic as scaler script
        drop_cols = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Label", "binary_label"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values
    
    del df
    gc.collect()
    return X, y, feature_cols

def main():
    print(f"Using device: {DEVICE}")
    
    # Load and scale data
    X, y, feature_cols = load_data()
    
    print("Scaling data...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Dataloaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Initialize Model
    input_dim = len(feature_cols)
    num_classes = 2
    model = create_lstm_torch(input_dim, num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                output = model(X_batch)
                pred = torch.argmax(output, dim=1)
                preds.extend(pred.cpu().numpy())
                labels.extend(y_batch.numpy())
        
        acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}, Test Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    # Final Eval and Report
    print(f"Training Complete. Best Acc: {best_acc:.4f}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    # (Same as before but for final report)
    preds = []
    labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            pred = torch.argmax(output, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y_batch.numpy())
            
    report = classification_report(labels, preds, target_names=["Benign", "Attack"])
    with open(RESULTS_PATH, "w") as f:
        f.write(f"LSTM CICIDS2017 Results\n{'='*50}\n")
        f.write(f"Test Accuracy: {accuracy_score(labels, preds)*100:.2f}%\n\n")
        f.write(report)
    
    print(f"Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
