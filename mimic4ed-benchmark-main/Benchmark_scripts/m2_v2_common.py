"""
M2 -- Shared utilities for the irregular-series (v2) pipeline
================================================================
Phase 3 of the session's M2 Q1-blockers plan: model training on the
cohort built by m2_build_irregular_cohort.py (Gate v2, signed off
2026-09-19). Mirrors m2_hybrid_common.py's role for the T=2 pipeline, but
for the variable-length (N, 11, 7) tensors -- kept as a SEPARATE module
rather than merged into m2_hybrid_common.py, so the T=2 pipeline (still
the source of the historical, now-archived 5-seed results) remains
untouched and independently reproducible from data/archive_pre_irregular/.

Delta-t convention: kept in RAW HOURS (not normalized by a reference
scale), matching Che et al. (2018)'s own formulation -- the learned
gamma_x_w/gamma_h parameters absorb whatever scale is appropriate via
w*delta+b before the relu/exp, so an external normalization is not
required by the paper and the T=2 pipeline's `ed_los_hours/median_LOS`
normalization was an implementation choice, not a modeling necessity.
Documented here explicitly since it is a genuine difference from the T=2
code path.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
VITALS_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
SCALER_PATH_V2 = os.path.join(DATA_DIR, "m2_v2_vitals_scaler.json")

# Horizon ablation (CLAUDE.md M2 required ablation: horizons 1h/6h/12h).
# Selected per PROCESS by the M2_V2_HORIZON environment variable; unset/empty
# means the composite `outcome_critical` target and every path, checkpoint name
# and result file is byte-for-byte what it was before this switch existed. A
# horizon run writes to parallel, suffixed locations so it can never overwrite
# or mix with the primary results. Scalers are deliberately NOT suffixed: both
# are fit on covariates of the dev-pool `training` subjects, never on the label.
HORIZON = os.environ.get("M2_V2_HORIZON", "").strip()
assert HORIZON in ("", "1h", "6h", "12h"), f"M2_V2_HORIZON must be one of '', 1h, 6h, 12h -- got {HORIZON!r}"
TARGET = f"y_{HORIZON}" if HORIZON else "outcome_critical"
SUFFIX = f"_hz{HORIZON}" if HORIZON else ""


def apply_horizon_filter(df):
    """Drops rows with no defined per-horizon label (horizon_eligible == False:
    54+7 encounters cohort-wide with no recoverable event_time, Gate v2 Sec 3).
    Those rows carry NaN in y_<h> BY DESIGN (fail-loud) and must be DROPPED, never
    coerced to 0. No-op for the composite target. The returned frame is a boolean
    view of its input, so it keeps the original index -- which is exactly the
    positional contract prepare_temporal_tensors_v2() relies on."""
    if not HORIZON:
        return df
    out = df[df["horizon_eligible"].astype(bool)]
    assert out[TARGET].notna().all(), f"{TARGET} has NaN on horizon-eligible rows -- cohort build is inconsistent."
    return out

sys.path.insert(0, SCRIPT_DIR)
from m2_grud_cell import GRUDCell, run_grud_masked
from m2_hybrid_common import StaticEncoder, FusionHead, DEVICE, CHECKPOINT_DIR

_SEQ_CACHE = {}


class GRUDWithEncoderV2(nn.Module):
    """Variable-length counterpart of m2_hybrid_common.GRUDWithEncoder:
    same GRUDCell + 2-layer classifier head, but driven by
    run_grud_masked() (which needs the extra pad_mask argument) instead of
    calling the cell directly. encode() exposes the raw hidden state for
    the same frozen-encoder pattern the T=2 pipeline uses (Stages 2/3),
    should the irregular-series pipeline later add a Frozen-Temporal
    Head Control / Hybrid GRU-D analogue."""

    def __init__(self, inp=7, hid=32, stat=0, out=1, clf_hidden=32, dropout=0.2):
        super().__init__()
        self.hid = hid
        self.cell = GRUDCell(inp=inp, hid=hid)
        self.clf = nn.Sequential(
            nn.Linear(hid + stat, clf_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(clf_hidden, out),
        )

    def encode(self, X, M, D, pad):
        return run_grud_masked(self.cell, X, M, D, pad)

    def forward(self, X, M, D, pad, S=None):
        h = self.encode(X, M, D, pad)
        if S is not None:
            h = torch.cat([h, S], 1)
        return self.clf(h)


class TemporalEncoderV2(nn.Module):
    """Frozen-encoder counterpart of m2_hybrid_common.TemporalEncoder, for
    the variable-length v2 recurrence -- holds just the GRUDCell (no
    classifier head), driven by run_grud_masked() (needs the extra pad
    argument TemporalEncoder's fixed-T=2 forward() never had)."""

    def __init__(self, inp=7, hid=32):
        super().__init__()
        self.hid = hid
        self.cell = GRUDCell(inp=inp, hid=hid)

    def forward(self, X, M, D, pad):
        return run_grud_masked(self.cell, X, M, D, pad)


def save_checkpoint_v2(model, name, seed):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{name}_v2{SUFFIX}_seed{seed}.pt"))


def load_temporal_only_model_v2(seed, hid=32):
    model = GRUDWithEncoderV2(inp=7, hid=hid, stat=0, clf_hidden=32).to(DEVICE)
    state = torch.load(os.path.join(CHECKPOINT_DIR, f"m2_temporal_only_grud_v2{SUFFIX}_seed{seed}.pt"),
                        map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_frozen_encoder_v2(seed, hid=32):
    """Loads seed's frozen v2 encoder from the Temporal-Only-v2 checkpoint,
    freezes its parameters, sets eval mode -- v2 counterpart of
    m2_hybrid_common.load_frozen_encoder(), same strict-load-verification
    discipline (never silently fall back to a randomly-initialised
    encoder)."""
    encoder = TemporalEncoderV2(inp=7, hid=hid).to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"m2_temporal_only_grud_v2{SUFFIX}_seed{seed}.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    missing, _unexpected = encoder.load_state_dict(state, strict=False)
    assert not missing, (
        f"Frozen v2 encoder checkpoint {ckpt_path} is missing {missing} -- "
        f"re-run m2_temporal_only_grud_v2.py (5-seed) before any downstream v2 stage. STOP."
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def load_fusion_head_v2(checkpoint_prefix, seed, temporal_dim=32, static_hidden=None,
                         static_dropout=0.2, fusion_dropout=0.1):
    static_encoder = (StaticEncoder(in_dim=4, hidden=static_hidden, dropout=static_dropout).to(DEVICE)
                       if static_hidden is not None else None)
    head = FusionHead(temporal_dim=temporal_dim, static_encoder=static_encoder,
                       fusion_dropout=fusion_dropout).to(DEVICE)
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_v2{SUFFIX}_seed{seed}.pt"), map_location=DEVICE)
    head.load_state_dict(ckpt)
    head.eval()
    return head


def predict_proba_temporal_only_v2(model, X, M, D, pad):
    model.eval()
    with torch.no_grad():
        logits = model(X, M, D, pad)
        return torch.sigmoid(logits).cpu().numpy().ravel()


def predict_proba_frozen_v2(encoder, head, X, M, D, pad, S=None):
    encoder.eval()
    head.eval()
    with torch.no_grad():
        h = encoder(X, M, D, pad)
        logits = head(h, S)
        return torch.sigmoid(logits).cpu().numpy().ravel()


# ── v2-specific result persistence (separate from the T=2 pipeline's
# m2_hybrid_predictions.csv / results_hybrid.csv, so the historical T=2
# result set is never mixed with or overwritten by v2 rows) ────────────────
V2_PREDICTIONS_PATH = os.path.join(DATA_DIR, f"m2_v2{SUFFIX}_predictions.csv")
V2_RESULTS_PATH = os.path.join(DATA_DIR, f"m2_v2{SUFFIX}_results.csv")


def save_predictions_v2(subject_id, model, seed, y_true, y_prob, stay_id=None):
    n = len(subject_id)
    rows = pd.DataFrame({
        "subject_id": subject_id, "stay_id": stay_id if stay_id is not None else [np.nan] * n,
        "model": model, "seed": seed, "y_true": y_true, "y_prob": y_prob,
    })
    if os.path.exists(V2_PREDICTIONS_PATH):
        existing = pd.read_csv(V2_PREDICTIONS_PATH)
        existing = existing[~((existing["model"] == model) & (existing["seed"] == seed))]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows.to_csv(V2_PREDICTIONS_PATH, index=False)


STATIC_SCALER_PATH_V2 = os.path.join(DATA_DIR, "m2_v2_static_scaler.json")
_STATIC_SCALER_V2 = None


def get_static_scaler_v2(force=False):
    """v2 counterpart of m2_hybrid_common.get_static_scaler(): mean/SD for
    HYBRID_STATIC_COLS, fit on the v2 dev-pool `training` subject subset
    only (the chronological-primary-split training period's oldest
    70/90). Kept separate from the T=2 scaler since the underlying subject
    population differs (chronological vs. random primary split)."""
    global _STATIC_SCALER_V2
    from m2_hybrid_common import HYBRID_STATIC_COLS, add_hybrid_static_features
    if _STATIC_SCALER_V2 is not None:
        return _STATIC_SCALER_V2
    if os.path.exists(STATIC_SCALER_PATH_V2) and not force:
        with open(STATIC_SCALER_PATH_V2) as f:
            _STATIC_SCALER_V2 = json.load(f)
            return _STATIC_SCALER_V2

    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    df = add_hybrid_static_features(load_split_frame("train_v2"))
    df = df[df["subject_id"].isin(train_ids)]
    vals = df[HYBRID_STATIC_COLS].values.astype(np.float64)
    mean = np.nanmean(vals, axis=0)
    sd = np.nanstd(vals, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    _STATIC_SCALER_V2 = {"cols": list(HYBRID_STATIC_COLS), "mean": mean.tolist(), "sd": sd.tolist()}
    with open(STATIC_SCALER_PATH_V2, "w") as f:
        json.dump(_STATIC_SCALER_V2, f, indent=2)
    return _STATIC_SCALER_V2


def prepare_static_features_v2(df):
    from m2_hybrid_common import HYBRID_STATIC_COLS
    sc = get_static_scaler_v2()
    assert sc["cols"] == list(HYBRID_STATIC_COLS), "v2 static scaler column order drift -- STOP."
    vals = df[HYBRID_STATIC_COLS].values.astype(np.float64)
    mean, sd = np.asarray(sc["mean"]), np.asarray(sc["sd"])
    return np.nan_to_num((vals - mean) / sd, nan=0.0).astype(np.float32)


def train_on_frozen_encoder_v2(encoder, X_fit, M_fit, D_fit, P_fit, y_fit, X_val, M_val, D_val, P_val, y_val,
                                seed, S_fit=None, S_val=None, static_hidden=None, static_dropout=0.2,
                                fusion_dropout=0.1, lr=1e-3, weight_decay=0.0,
                                batch_size=128, max_epochs=100, patience=10):
    """v2 (variable-length) counterpart of m2_hybrid_common.train_on_frozen_
    encoder(): identical design (precompute the frozen encoder's embedding
    ONCE since it never changes during head training, then train only the
    StaticEncoder+FusionHead on those fixed embeddings via the same shared
    _minibatch_early_stop_loop) -- the only change is passing `pad` through
    the encoder call, since TemporalEncoderV2 needs it and TemporalEncoder
    did not."""
    from m2_hybrid_common import _minibatch_early_stop_loop, DEVICE as _DEVICE

    encoder.eval()
    with torch.no_grad():
        h_fit = encoder(X_fit, M_fit, D_fit, P_fit).detach()
        h_val = encoder(X_val, M_val, D_val, P_val).detach()

    static_encoder = None
    s_fit_t = s_val_t = None
    if static_hidden is not None:
        torch.manual_seed(seed)
        static_encoder = StaticEncoder(in_dim=S_fit.shape[1], hidden=static_hidden, dropout=static_dropout).to(_DEVICE)
        s_fit_t = S_fit
        s_val_t = S_val

    head = FusionHead(temporal_dim=h_fit.shape[1], static_encoder=static_encoder,
                       fusion_dropout=fusion_dropout).to(_DEVICE)

    X_fit_t = [h_fit] + ([s_fit_t] if s_fit_t is not None else [])
    X_val_t = [h_val] + ([s_val_t] if s_val_t is not None else [])

    def forward_fn(ts):
        return head(ts[0], ts[1] if len(ts) > 1 else None)

    history = _minibatch_early_stop_loop(
        param_groups_fn=lambda: head.parameters(),
        forward_fn=forward_fn,
        X_fit_tensors=X_fit_t, y_fit=y_fit, X_val_tensors=X_val_t, y_val=y_val,
        seed=seed, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, train_modules=[head],
    )
    return head, history


def save_seed_metrics_v2(model, seed, metrics: dict):
    row = {"model": model, "seed": seed, **metrics}
    if os.path.exists(V2_RESULTS_PATH):
        existing = pd.read_csv(V2_RESULTS_PATH)
        existing = existing[~((existing["model"] == model) & (existing["seed"] == seed))]
        rows = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        rows = pd.DataFrame([row])
    rows.to_csv(V2_RESULTS_PATH, index=False)


def load_split_frame(split_name):
    """split_name: 'train_v2' or 'test_v2'."""
    return pd.read_csv(os.path.join(DATA_DIR, f"{split_name}.csv"))


def load_sequences(split_name):
    """Returns (values, obs_mask, pad_mask, delta_hours, stay_id) for the
    given split, cached per-process (each array can be ~100s of MB)."""
    if split_name in _SEQ_CACHE:
        return _SEQ_CACHE[split_name]
    path = os.path.join(DATA_DIR, f"m2_irregular_sequences_{split_name}.npz")
    with np.load(path) as z:
        out = (z["values"], z["obs_mask"], z["pad_mask"], z["delta_hours"], z["stay_id"])
    _SEQ_CACHE[split_name] = out
    return out


def load_dev_pool_split_v2():
    with open(os.path.join(DATA_DIR, "m2_v2_dev_pool_split.json")) as f:
        saved = json.load(f)
    return {k: {"subject_ids": np.array(v["subject_ids"]), "sha256": v["sha256"]}
            for k, v in saved.items()}


def get_vitals_scaler_v2(force=False):
    """Per-vital mean/SD, fit on the v2 dev-pool `training` subject subset
    ONLY, on the REAL (non-padded, non-missing) observations across every
    timestep -- not just triage+last, since every genuine observation in
    the irregular sequence now carries information the T=2 scaler never
    saw. Missing/padded entries remain excluded from the fit via obs_mask,
    exactly as get_vitals_scaler() does for the T=2 pipeline."""
    if os.path.exists(SCALER_PATH_V2) and not force:
        with open(SCALER_PATH_V2) as f:
            return json.load(f)

    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    df = load_split_frame("train_v2")
    values, obs_mask, pad_mask, delta_hours, stay_id = load_sequences("train_v2")

    row_mask = df["subject_id"].isin(train_ids).values
    v = values[row_mask]      # (n, 11, 7)
    m = obs_mask[row_mask].astype(bool)

    mean, sd = {}, {}
    for i, c in enumerate(VITALS_COLS):
        vals = v[:, :, i][m[:, :, i]]
        mean[c] = float(np.mean(vals)) if len(vals) else 0.0
        s = float(np.std(vals)) if len(vals) else 1.0
        sd[c] = s if s > 1e-8 else 1.0

    scaler = {"version": 1, "mean": mean, "sd": sd,
              "fit_on": "v2 dev-pool `training` subject subset only, all real observations",
              "n_train_rows": int(row_mask.sum())}
    with open(SCALER_PATH_V2, "w") as f:
        json.dump(scaler, f, indent=2)
    return scaler


def prepare_temporal_tensors_v2(df, split_name, device="cpu"):
    """Returns torch tensors (X, M, D, pad, y), each (N, 11, 7) except pad
    (N, 11) and y (N,), for the rows of `df` in df's own row order. Each
    row's sequence is looked up by its stay_id in
    m2_irregular_sequences_{split_name}.npz, so `df` may be filtered,
    re-ordered (e.g. sort_chronologically) or re-indexed freely.

    Standardizes real observations using get_vitals_scaler_v2(); missing
    or padded entries become 0.0 in scaled space (== training mean, GRU-D's
    x_tilde), matching the T=2 pipeline's convention.
    """
    full_df = load_split_frame(split_name)
    values, obs_mask, pad_mask, delta_hours, stay_id = load_sequences(split_name)
    assert len(full_df) == len(values), f"{split_name}.csv and its npz have diverged in row count"

    # Rows are selected by stay_id, NEVER by df.index (Deviation 23, 2026-10-01). The previous version
    # used df.index.values as the row position and only asserted it was in range; sort_chronologically()
    # ends in reset_index(drop=True), so every conformal script that sorted its frame first got the
    # sequence tensors of rows 0..n-1 of the split file paired with the sorted frame's labels and static
    # features (Hybrid GRU-D v2 test AUROC 0.9375 -> 0.7324). Looking rows up by stay_id makes the result
    # independent of the frame's order and index.
    seq_pos = pd.Series(np.arange(len(stay_id)), index=stay_id)
    assert seq_pos.index.is_unique, f"{split_name}: stay_id is not unique in the sequence file"
    missing = ~pd.Index(df["stay_id"].values).isin(seq_pos.index)
    assert not missing.any(), f"{split_name}: {int(missing.sum())} stay_id(s) in df have no stored sequence"
    pos = seq_pos.loc[df["stay_id"].values].values

    X = values[pos]
    M = obs_mask[pos]
    D = delta_hours[pos]
    P = pad_mask[pos]

    sc = get_vitals_scaler_v2()
    mu = np.array([sc["mean"][c] for c in VITALS_COLS], dtype=np.float64)
    sd = np.array([sc["sd"][c] for c in VITALS_COLS], dtype=np.float64)
    X_scaled = np.where(M.astype(bool), (X - mu) / sd, 0.0)

    y = df[TARGET].astype(float).values

    return (torch.tensor(X_scaled, dtype=torch.float32, device=device),
            torch.tensor(M, dtype=torch.float32, device=device),
            torch.tensor(D, dtype=torch.float32, device=device),
            torch.tensor(P, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))
