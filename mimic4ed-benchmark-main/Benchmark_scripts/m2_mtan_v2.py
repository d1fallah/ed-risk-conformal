"""
M2 -- mTAN comparator, irregular-series (v2) cohort (Phase 5, Gate v2 Sec 8)
==============================================================================
Vendored `reml-lab/mTAN` (Shukla & Marlin, "Multi-Time Attention Networks
for Irregularly Sampled Time Series," ICLR 2021), commit 7a3d536e, cloned
into module2/mTAN/ per CLAUDE.md Sec 1.7 -- imports `enc_mtan_classif`
directly from the vendored src/models.py rather than reimplementing the
attention/time-embedding mechanism.

Single-model, single-seed run FIRST, matching the session plan's own rule
(measure before committing to a full 5-seed run). Uses the SAME 4-way
dev-pool split, SAME batch size (128) and SAME early-stopping protocol
(_minibatch_early_stop_loop, patience=10, max_epochs=100) as
m2_temporal_only_grud_v2.py, so the comparison is under a matched
protocol, not an unmatched one favoring either architecture.

Input construction for enc_mtan_classif(x, time_steps):
  x           = concat([scaled_values, obs_mask], dim=-1)  shape (B, 11, 14)
  time_steps  = t_hours / 24.0                              shape (B, 11)
                (rescaled to ~[0, few] range; mTAN's own reference
                implementation scales its query grid the same way via
                `48.*pos` inside time_embedding/learn_time_embedding, i.e.
                it expects pos in a small, roughly-day-scale range, not
                raw hours -- passing raw hours (up to ~74 in this cohort,
                see M2_Outcome_Definition_Gate.md Sec 4) would blow that
                internal scaling out of its intended regime)
  query       = torch.linspace(0, 1, num_ref_points) -- a fixed reference
                time grid the attention mechanism re-samples onto,
                independent of any one sequence's actual observation
                times (mTAN's whole mechanism: attend from irregular
                observed times onto this fixed reference grid).

Padding: mTAN's masked-attention (multiTimeAttention) already handles
irregular/missing observations via the same style of binary mask GRU-D
uses (the `mask` argument replicated as [value_mask, value_mask] inside
enc_mtan_classif.forward), so no separate variable-length driver is
needed here the way GRUDCell needed run_grud_masked() -- a fully-padded
(all-mask-zero) timestep simply contributes ~nothing through the
attention-weighted aggregation, which is the intended behavior, not a bug
requiring a workaround.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MTAN_SRC = os.path.join(SCRIPT_DIR, "..", "..", "mTAN", "src")
SEED_DIR = os.path.join(DATA_DIR, "m2_v2_seed_results", "mtan")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, MTAN_SRC)
from models import enc_mtan_classif  # vendored, reml-lab/mTAN commit 7a3d536e
from m2_hybrid_common import (
    _minibatch_early_stop_loop, expected_calibration_error, calibration_slope_intercept, DEVICE,
)
from m2_v2_common import load_split_frame, load_dev_pool_split_v2, VITALS_COLS, get_vitals_scaler_v2

HIDDEN, WEIGHT_DECAY, LR, BATCH_SIZE = 32, 1e-4, 1e-3, 128
EMBED_TIME, NUM_HEADS, NUM_REF_POINTS = 32, 1, 16
MAX_EPOCHS = 100
PATIENCE = 10
TIME_SCALE_HOURS = 24.0  # rescales raw hours to mTAN's expected small-range input


class MTANClassifierHead(nn.Module):
    """enc_mtan_classif already ends in its own 2-class softmax classifier
    (models.py L192-197) -- this wrapper just exposes a single logit
    (class-1 minus class-0) so the SAME _minibatch_early_stop_loop /
    BCEWithLogitsLoss infrastructure GRU-D v2 uses can drive this model
    too, without a second training-loop implementation."""

    def __init__(self, input_dim, query, nhidden, embed_time, num_heads, device):
        super().__init__()
        self.enc = enc_mtan_classif(
            input_dim=input_dim, query=query, nhidden=nhidden,
            embed_time=embed_time, num_heads=num_heads, learn_emb=True, device=device,
        )

    def forward(self, x, time_steps):
        logits2 = self.enc(x, time_steps)  # (B, 2)
        return (logits2[:, 1] - logits2[:, 0]).unsqueeze(1)  # (B, 1), BCEWithLogits-compatible


def prepare_mtan_tensors(df, split_name, device):
    full_df = load_split_frame(split_name)
    with np.load(os.path.join(DATA_DIR, f"m2_irregular_sequences_{split_name}.npz")) as z:
        values, obs_mask, t_hours = z["values"], z["obs_mask"], z["t_hours"]
    assert len(full_df) == len(values), f"{split_name}.csv/npz row-count mismatch"
    pos = df.index.values

    v, m, th = values[pos], obs_mask[pos], t_hours[pos]
    sc = get_vitals_scaler_v2()
    mu = np.array([sc["mean"][c] for c in VITALS_COLS])
    sd = np.array([sc["sd"][c] for c in VITALS_COLS])
    v_scaled = np.where(m.astype(bool), (v - mu) / sd, 0.0)

    x = np.concatenate([v_scaled, m], axis=-1)  # (N, 11, 14)
    time_steps = th / TIME_SCALE_HOURS
    y = df["outcome_critical"].astype(float).values

    return (torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(time_steps, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def main():
    print(f"Device: {DEVICE}")
    df_train = load_split_frame("train_v2")
    df_test = load_split_frame("test_v2")
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())

    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]

    t0 = time.perf_counter()
    Xf, Tf, yf = prepare_mtan_tensors(df_fit, "train_v2", DEVICE)
    Xv, Tv, yv = prepare_mtan_tensors(df_val, "train_v2", DEVICE)
    Xt, Tt, yt = prepare_mtan_tensors(df_test, "test_v2", DEVICE)
    t_data = time.perf_counter() - t0
    print(f"Data prep: {t_data:.1f}s  (fit={len(df_fit)}, val={len(df_val)}, test={len(df_test)})")

    seed = 42
    torch.manual_seed(seed)
    query = torch.linspace(0, 1, NUM_REF_POINTS)
    model = MTANClassifierHead(
        input_dim=7, query=query, nhidden=HIDDEN, embed_time=EMBED_TIME,
        num_heads=NUM_HEADS, device=DEVICE,
    ).to(DEVICE)

    t0 = time.perf_counter()
    history = _minibatch_early_stop_loop(
        param_groups_fn=lambda: model.parameters(),
        forward_fn=lambda ts: model(ts[0], ts[1]),
        X_fit_tensors=[Xf, Tf], y_fit=yf.cpu().numpy(),
        X_val_tensors=[Xv, Tv], y_val=yv.cpu().numpy(),
        seed=seed, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, train_modules=[model],
    )
    t_train = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        test_logits = model(Xt, Tt)
        test_prob = torch.sigmoid(test_logits).cpu().numpy().ravel()
    y_test_np = yt.cpu().numpy()

    auroc = roc_auc_score(y_test_np, test_prob)
    auprc = average_precision_score(y_test_np, test_prob)
    brier = brier_score_loss(y_test_np, test_prob)
    ece = expected_calibration_error(y_test_np, test_prob)
    slope, intercept = calibration_slope_intercept(y_test_np, test_prob)

    metrics = {
        "seed": seed, "epochs_trained": len(history), "stopped_early": len(history) < MAX_EPOCHS,
        "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
        "ece": float(ece), "cal_slope": float(slope), "cal_intercept": float(intercept),
        "data_prep_seconds": t_data, "train_seconds": t_train,
        "config": {"hidden": HIDDEN, "embed_time": EMBED_TIME, "num_heads": NUM_HEADS,
                   "num_ref_points": NUM_REF_POINTS, "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
    }
    with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== mTAN, single-model single-seed (seed={seed}) result ===")
    print(f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")
    print(f"epochs_trained={len(history)} (stopped_early={len(history) < MAX_EPOCHS})")
    print(f"data_prep={t_data:.1f}s  train={t_train:.1f}s")


if __name__ == "__main__":
    main()
