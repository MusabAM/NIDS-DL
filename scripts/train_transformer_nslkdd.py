import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import time

# Add parent directory to path to import src and backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classical.transformer import TransformerClassifier
from backend.utils import NSL_KDD_COLUMNS

# Configuration
DATA_PATH = 'data/raw/nsl-kdd/'
MODEL_SAVE_PATH = 'results/models/final prd models/transformer_nsl_kdd.pth'
SCALER_SAVE_PATH = 'results/models/final prd models/transformer_scaler.pkl'
FEATURES_SAVE_PATH = 'results/models/final prd models/transformer_features.pkl'

# Hyperparameters
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_BLOCKS = 3
DENSE_UNITS = [64]
DROPOUT = 0.3
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.1

def load_and_preprocess():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.txt'), header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.txt'), header=None, names=NSL_KDD_COLUMNS)

    # Convert labels to binary (normal=0, others=1)
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    # One-hot encoding
    categorical_cols = ["protocol_type", "service", "flag"]
    train_encoded = pd.get_dummies(train_df, columns=categorical_cols)
    test_encoded = pd.get_dummies(test_df, columns=categorical_cols)

    # Align columns
    target_cols = [col for col in train_encoded.columns if col not in ['label', 'difficulty_level']]
    
    # Add missing columns to test set
    for col in target_cols:
        if col not in test_encoded.columns:
            test_encoded[col] = 0
            
    # Keep only the target columns in correct order
    X_train = train_encoded[target_cols].values.astype(np.float32)
    y_train = train_encoded['label'].values
    X_test = test_encoded[target_cols].values.astype(np.float32)
    y_test = test_encoded['label'].values

    print(f"Features after encoding: {X_train.shape[1]}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature list
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(FEATURES_SAVE_PATH, 'wb') as f:
        pickle.dump(target_cols, f)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    return X_train_scaled, y_train, X_test_scaled, y_test, X_train.shape[1]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train, y_train, X_test, y_test, input_dim = load_and_preprocess()

    # Create DataLoaders
    # Validation split
    split_idx = int(len(X_train) * (1 - VAL_SPLIT))
    X_train_part, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_part, y_val = y_train[:split_idx], y_train[split_idx:]

    train_ds = TensorDataset(torch.FloatTensor(X_train_part), torch.LongTensor(y_train_part))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Initialize model
    model = TransformerClassifier(
        input_dim=input_dim,
        num_classes=2,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_blocks=NUM_BLOCKS,
        dense_units=DENSE_UNITS,
        dropout=DROPOUT
    ).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")

    # Final evaluation on test set
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()
    
    print(f"Final Test Accuracy: {test_correct / test_total:.4f}")

if __name__ == "__main__":
    train()
