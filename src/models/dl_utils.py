"""
Shared helpers for LSTM & Transformer training:
  - Sequence/window builder
  - PyTorch Dataset
  - Training loop with early stopping
  - Volatility target construction
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Target construction

def build_volatility_target(returns, target_window=5):
    """
    Build the volatility target the models will learn to predict.
    Uses forward-looking realised volatility over `target_window` days,
    i.e. the std of the NEXT `target_window` returns.

    Returns a numpy array aligned with `returns` (NaN where target unavailable).
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    target = np.full(n, np.nan)
    for i in range(n - target_window):
        target[i] = np.std(returns[i + 1 : i + 1 + target_window], ddof=0)
    return target

# Sequence builder (sliding window)

def create_sequences(features, target, seq_len):
    """
    Convert a 1D/2D feature array + target into supervised sequences.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    target = np.asarray(target, dtype=float)

    X, y, idx = [], [], []
    for i in range(seq_len, len(features)):
        if np.isnan(target[i]):
            continue
        window = features[i - seq_len : i]
        if np.isnan(window).any():
            continue
        X.append(window)
        y.append(target[i])
        idx.append(i)
    if not X:
        return (np.empty((0, seq_len, features.shape[1])),
                np.empty((0,)), np.empty((0,), dtype=int))
    return np.array(X), np.array(y), np.array(idx)

# PyTorch Dataset

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Training loop with early stopping

def train_model(
    model, train_loader, val_loader,
    epochs=50, lr=1e-3, patience=8,
    device="cpu", verbose=True,
):
    """
    Train a model with Adam + MSE loss + early stopping on validation loss.
    Returns (trained_model, history_dict).
    """
    model    = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    history    = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"      Epoch {epoch:3d}/{epochs}  train={train_loss:.6e}  val={val_loss:.6e}")

        # Early stopping
        if val_loss < best_val - 1e-9:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

# Inference

def predict(model, X, device="cpu", batch_size=256):
    """Run inference, return 1D numpy array of predictions."""
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds).flatten()