"""
dl_baseline_bridge.py
=====================
Bridge: mimic4ed-benchmark-main/data/{train,test}.csv
     -> Deep-learning baselines (MLP, LSTM, GRU)
        modelled after CAREBench (transformer/trainer.py + classic/trainer.py)
        and MDS-ED (clinical_ts tabular baselines)

Task: Critical Outcome Prediction (Task 2)

Outputs AUROC + AUPRC, then appends rows to
mimic4ed-benchmark-main/data/baseline_results.csv
under category "Deep Learning Baselines".

Architecture notes
------------------
MLP  -- 3-layer feed-forward; mirrors CAREBench classic tabular baseline MLP
LSTM -- single-layer LSTM encoding a 2-step vital-sign sequence; mirrors
        MDS-ED tabular+temporal baseline
GRU  -- same but with GRU cell; mirrors CAREBench transformer trainer GRU
        variant used for EHR decompensation
"""

import os, sys, warnings, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_CSV  = os.path.join(DATA_DIR, "baseline_results.csv")
TRAIN_PATH   = os.path.join(DATA_DIR, "train.csv")
TEST_PATH    = os.path.join(DATA_DIR, "test.csv")

TARGET_COL   = "outcome_critical"
N_EPOCHS     = 120
BATCH_SIZE   = 16
LR           = 1e-3
DEVICE       = torch.device("cpu")   # CPU-only; demo dataset is tiny

# ── Feature sets (same as run_baseline_models.py Task 2) ─────────────────────
VITALS       = ['temperature','heartrate','resprate','o2sat','sbp','dbp','pain']
TRIAGE_COLS  = [f'triage_{v}' for v in VITALS]
ED_COLS      = [f'ed_{v}_last' for v in VITALS]

STATIC_COLS  = [
    "age", "gender", "triage_acuity",
    "n_ed_30d","n_ed_90d","n_ed_365d",
    "n_hosp_30d","n_hosp_90d","n_hosp_365d",
    "n_icu_30d","n_icu_90d","n_icu_365d",
    "chiefcom_chest_pain","chiefcom_abdominal_pain","chiefcom_headache",
    "chiefcom_shortness_of_breath","chiefcom_back_pain","chiefcom_cough",
    "chiefcom_nausea_vomiting","chiefcom_fever_chills",
    "chiefcom_syncope","chiefcom_dizziness",
    "n_med","n_medrecon",
]

# ── Data helpers ──────────────────────────────────────────────────────────────
def load_flat(df):
    """Return flat feature matrix X (N, D) and label y (N,)."""
    df = df.copy()
    df['gender'] = df['gender'].map({'M':1,'F':0,'m':1,'f':0}).fillna(0).astype(float)
    all_cols = TRIAGE_COLS + ED_COLS + STATIC_COLS
    missing  = [c for c in all_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    X = df[all_cols].fillna(df[all_cols].median(numeric_only=True)).fillna(0).values.astype(np.float32)
    y = df[TARGET_COL].astype(int).values
    return X, y

def load_seq(df):
    """Return sequence tensor X (N, T=2, D=7) and label y (N,)."""
    df = df.copy()
    df['gender'] = df['gender'].map({'M':1,'F':0,'m':1,'f':0}).fillna(0).astype(float)
    for c in TRIAGE_COLS + ED_COLS:
        if c not in df.columns:
            df[c] = 0.0
    T = df[TRIAGE_COLS].fillna(0).values.astype(np.float32)
    E = df[ED_COLS].fillna(0).values.astype(np.float32)
    X = np.stack([T, E], axis=1)          # (N, 2, 7)
    y = df[TARGET_COL].astype(int).values
    return X, y

def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

# ── Model Definitions ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    """3-layer MLP — mirrors CAREBench classic tabular baseline architecture."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class LSTMNet(nn.Module):
    """Single-layer LSTM over triage→ED vital-sign sequence.
    Mirrors MDS-ED clinical_ts tabular+temporal baselines."""
    def __init__(self, input_size=7, hidden_size=32, static_size=24):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.static_size = static_size

    def forward(self, x_seq, x_static):
        _, (h, _) = self.lstm(x_seq)
        h = h.squeeze(0)
        return self.head(torch.cat([h, x_static], dim=1))

class GRUNet(nn.Module):
    """Single-layer GRU over triage→ED vital-sign sequence.
    Mirrors CAREBench transformer trainer GRU variant for EHR decompensation."""
    def __init__(self, input_size=7, hidden_size=32, static_size=24):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.static_size = static_size

    def forward(self, x_seq, x_static):
        _, h = self.gru(x_seq)
        h = h.squeeze(0)
        return self.head(torch.cat([h, x_static], dim=1))

# ── Generic Trainer ────────────────────────────────────────────────────────────
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        if len(batch) == 2:
            x, y = batch
            x, y = x.to(DEVICE), y.unsqueeze(1).to(DEVICE)
            out  = model(x)
        else:
            x_seq, x_static, y = batch
            x_seq, x_static, y = x_seq.to(DEVICE), x_static.to(DEVICE), y.unsqueeze(1).to(DEVICE)
            out = model(x_seq, x_static)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        if len(batch) == 2:
            x, y = batch
            x    = x.to(DEVICE)
            out  = model(x)
        else:
            x_seq, x_static, y = batch
            x_seq, x_static = x_seq.to(DEVICE), x_static.to(DEVICE)
            out = model(x_seq, x_static)
        probs = torch.sigmoid(out).squeeze(1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_probs)

def run_experiment(model, train_loader, test_loader, tag):
    # Compute class balance directly from the underlying TensorDataset
    all_y = train_loader.dataset.tensors[-1].numpy()
    pos   = int(all_y.sum())
    neg   = len(all_y) - pos
    pw   = torch.tensor([neg / max(int(pos), 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_auroc, best_auprc = 0, 0
    for epoch in range(1, N_EPOCHS + 1):
        train_model(model, train_loader, optimizer, criterion)
        if epoch % 20 == 0 or epoch == N_EPOCHS:
            y_true, y_prob = evaluate(model, test_loader)
            if y_true.sum() > 0:
                auroc = roc_auc_score(y_true, y_prob)
                auprc = average_precision_score(y_true, y_prob)
                if auroc > best_auroc:
                    best_auroc, best_auprc = auroc, auprc

    print(f"  {tag:<12}  AUROC={best_auroc:.4f}  AUPRC={best_auprc:.4f}")
    return best_auroc, best_auprc

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Deep Learning Baselines — Task 2: Critical Outcome")
    print("Architectures sourced from CAREBench / MDS-ED patterns")
    print("=" * 60)

    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    print(f"Train: {df_train.shape}  |  Test: {df_test.shape}")
    print(f"Target positive rate  train={df_train[TARGET_COL].mean():.1%}  "
          f"test={df_test[TARGET_COL].mean():.1%}\n")

    # ---- Flat features (for MLP) ----
    X_flat_train, y_train_flat = load_flat(df_train)
    X_flat_test,  y_test_flat  = load_flat(df_test)

    scaler_flat = StandardScaler()
    X_flat_train = scaler_flat.fit_transform(X_flat_train)
    X_flat_test  = scaler_flat.transform(X_flat_test)

    mlp_train_loader = make_loader(X_flat_train.astype(np.float32), y_train_flat)
    mlp_test_loader  = make_loader(X_flat_test.astype(np.float32),  y_test_flat, shuffle=False)

    # ---- Sequence features (for LSTM / GRU) ----
    X_seq_train, y_train_seq = load_seq(df_train)
    X_seq_test,  y_test_seq  = load_seq(df_test)

    # Static: everything except the vital time-series
    flat_all_cols   = TRIAGE_COLS + ED_COLS + STATIC_COLS
    X_static_train, _ = load_flat(df_train)
    X_static_test,  _ = load_flat(df_test)
    X_static_train  = scaler_flat.transform(
        scaler_flat.inverse_transform(X_static_train))   # already scaled above
    X_static_test   = scaler_flat.transform(
        scaler_flat.inverse_transform(X_static_test))

    static_size = X_static_train.shape[1]

    # Sequence scaler (per-feature over T dimension)
    N, T, D = X_seq_train.shape
    seq_scaler = StandardScaler()
    X_seq_train_sc = seq_scaler.fit_transform(X_seq_train.reshape(-1, D)).reshape(N, T, D).astype(np.float32)
    Nt, _, _ = X_seq_test.shape
    X_seq_test_sc  = seq_scaler.transform(X_seq_test.reshape(-1, D)).reshape(Nt, T, D).astype(np.float32)

    # Combined DataLoaders for LSTM/GRU: (seq, static, label)
    def make_rnn_loader(X_seq, X_static, y, shuffle=True):
        ds = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(X_static, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    rnn_train_loader = make_rnn_loader(X_seq_train_sc, X_static_train.astype(np.float32), y_train_seq)
    rnn_test_loader  = make_rnn_loader(X_seq_test_sc,  X_static_test.astype(np.float32),  y_test_seq,  shuffle=False)

    # ---- Run experiments ----
    results = []

    # 1. MLP (CAREBench classic baseline style)
    mlp = MLP(input_dim=X_flat_train.shape[1]).to(DEVICE)
    auroc, auprc = run_experiment(mlp, mlp_train_loader, mlp_test_loader, "MLP")
    results.append(("MLP (CAREBench)", auroc, auprc))

    # 2. LSTM (MDS-ED tabular+temporal baseline style)
    lstm = LSTMNet(input_size=7, hidden_size=32, static_size=static_size).to(DEVICE)
    auroc, auprc = run_experiment(lstm, rnn_train_loader, rnn_test_loader, "LSTM")
    results.append(("LSTM (MDS-ED)", auroc, auprc))

    # 3. GRU (CAREBench transformer-trainer GRU style)
    gru = GRUNet(input_size=7, hidden_size=32, static_size=static_size).to(DEVICE)
    auroc, auprc = run_experiment(gru, rnn_train_loader, rnn_test_loader, "GRU")
    results.append(("GRU (CAREBench)", auroc, auprc))

    # ---- Append to baseline_results.csv ----
    new_rows = []
    for model_name, auroc, auprc in results:
        new_rows.append({
            "task":           "Task 2 - Critical Outcome",
            "target":         TARGET_COL,
            "model":          model_name,
            "auroc":          round(auroc, 4),
            "auprc":          round(auprc, 4),
            "n_train":        len(df_train),
            "n_test":         len(df_test),
            "pos_rate_test":  round(df_test[TARGET_COL].mean(), 4),
            "category":       "Deep Learning Baselines",
        })

    df_new = pd.DataFrame(new_rows)

    if os.path.exists(RESULTS_CSV):
        df_existing = pd.read_csv(RESULTS_CSV)
        # Add category column to existing rows if missing
        if "category" not in df_existing.columns:
            df_existing["category"] = "Classical Baselines"
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 60)
    print("Summary — Deep Learning Baselines (Task 2)")
    print("=" * 60)
    print(f"  {'Model':<22} {'AUROC':>8}  {'AUPRC':>8}")
    print(f"  {'-'*22} {'-'*8}  {'-'*8}")
    for model_name, auroc, auprc in results:
        print(f"  {model_name:<22} {auroc:>8.4f}  {auprc:>8.4f}")
    print(f"\nResults appended -> {RESULTS_CSV}")
    print("\nFull updated table:")
    print(df_combined[["task","model","auroc","auprc","category"]].to_string(index=False))

if __name__ == "__main__":
    main()
