"""
Shared utilities for the M2 Hybrid GRU-D controlled experiment. Originally
written against M2_Hybrid_GRU_D_Student_Execution_Guide_v3_2.docx; the
static-feature-confound/fusion-width/learning-rate/historical-baseline
questions raised against that version were resolved by the supervisor's
M2_Hybrid_GRU_D_Student_Execution_Guide_v3_8_EXECUTION_READY.docx (see
M2_Hybrid_Guide_Clarification_Request.md for the questions asked and
M2_Findings_Dossier.md for how each was resolved) -- this module now also
implements v3.8's fixed 4-way development-pool split and reproducibility
manifest (Sec 5.1/Sec 7/"Implementation Fingerprint").

Reuses split/conformal helpers from m2_ablations.py rather than duplicating
them (patient_level_split, _sorted_bounds, run_mapie, run_eci,
GRUDRiskRegressor). Adds only what doesn't already exist in the M2 pipeline:
the 4-static-feature construction, the fixed dev-pool split, a
reproducibility-manifest writer, and per-patient prediction-saving (needed
for the Guide's patient-cluster bootstrap, which requires paired
per-patient predictions across models -- no prior M2 script saves anything
below aggregate metrics).
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")

# Re-exported, not reimplemented. See M2_Findings_Dossier.md Sec 1.2 for why
# patient-level (not row-level) splitting is mandatory: 34% of patients have
# >1 ED visit, and a row-level split leaks the same patient across sides
# (confirmed: 10,156 overlapping patients at seed=42 before this fix).
from m2_ablations import (  # noqa: F401
    patient_level_split, _sorted_bounds, run_mapie, run_eci, GRUDRiskRegressor,
)

TARGET = "outcome_critical"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determinism pin (cheap insurance; cf. M4's cuDNN non-determinism finding).
# GRUDWithEncoder/TemporalEncoder implement the recurrence by hand via
# nn.Linear + elementwise ops in a Python loop, not nn.GRU/nn.LSTM, so
# cuDNN's fused-RNN-kernel non-determinism (M4's actual root cause) does
# not apply the same way here -- this pin is defensive, not a fix for a
# confirmed issue in this module.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -- Temporal (vitals-only) tensor construction, shared by all 3 GRU-D-family
# models (Temporal-Only GRU-D, Frozen-Temporal Head Control, Hybrid GRU-D) --
VITALS_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]

# Byte-identical to data_general_processing.py's own table (itself inherited
# from mimic-extract). Duplicated here rather than imported because
# data_general_processing.py is an argparse SCRIPT that parses required CLI
# args at import time -- importing it from a library module would break every
# caller. Guarded by _assert_valid_range_matches_pipeline() below, so the two
# copies can never silently drift.
VITALS_VALID_RANGE = {
    "temperature": {"outlier_low": 14.2, "valid_low": 26, "valid_high": 45, "outlier_high": 47},
    "heartrate": {"outlier_low": 0, "valid_low": 0, "valid_high": 350, "outlier_high": 390},
    "resprate": {"outlier_low": 0, "valid_low": 0, "valid_high": 300, "outlier_high": 330},
    "o2sat": {"outlier_low": 0, "valid_low": 0, "valid_high": 100, "outlier_high": 150},
    "sbp": {"outlier_low": 0, "valid_low": 0, "valid_high": 375, "outlier_high": 375},
    "dbp": {"outlier_low": 0, "valid_low": 0, "valid_high": 375, "outlier_high": 375},
    "pain": {"outlier_low": 0, "valid_low": 0, "valid_high": 10, "outlier_high": 10},
    "acuity": {"outlier_low": 1, "valid_low": 1, "valid_high": 5, "outlier_high": 5},
}

SCALER_VERSION = 2  # bump to invalidate data/m2_vitals_scaler.json

_MASTER_VITALS_LOOKUP = None  # module-level cache; master_dataset.csv is ~226MB
_VITALS_SCALER = None
_SCALER_PATH = os.path.join(DATA_DIR, "m2_vitals_scaler.json")


def _assert_valid_range_matches_pipeline():
    """Fails loudly if VITALS_VALID_RANGE above has drifted from the table
    data_general_processing.py actually used to clean train.csv/test.csv.
    Parsed textually (not imported) for the reason given above."""
    src = os.path.join(SCRIPT_DIR, "data_general_processing.py")
    with open(src) as f:
        text = f.read()
    start = text.index("vitals_valid_range = {")
    end = text.index("\n}\n", start) + 3
    ns = {}
    exec(text[start:end], ns)  # noqa: S102 - trusted, in-repo source
    assert ns["vitals_valid_range"] == VITALS_VALID_RANGE, (
        "VITALS_VALID_RANGE has drifted from data_general_processing.py's table -- "
        "the temporal models would be cleaned differently from train.csv/test.csv. STOP."
    )


def _clean_vitals_frame(df_vitals):
    """Applies EXACTLY the cleaning data_general_processing.py applies to
    master_dataset.csv before train.csv/test.csv are written, in the same
    order: convert_temp_to_celcius() then remove_outliers().

    Why this exists (M2_Hybrid_Deviation_Log.md Deviation 7): Deviation 4's
    fix moved the temporal pipeline's vitals SOURCE from train.csv/test.csv
    to master_dataset.csv, to recover genuine pre-imputation missingness.
    But master_dataset.csv is the stage BEFORE the Fahrenheit->Celsius
    conversion and BEFORE outlier removal, so that fix silently reverted
    both -- verified directly on the frozen test cohort: triage_temperature
    arrived in Fahrenheit (mean 98.0, max 986.0), triage_o2sat reached
    9322, triage_sbp reached 151103. Those values went unscaled into a
    tanh/sigmoid GRU. Cleaning here restores parity with train.csv/test.csv
    while KEEPING the NaNs (the point of Deviation 4's fix) instead of
    median-imputing them.

    remove_outliers is two-tier and both tiers matter: values outside
    [outlier_low, outlier_high] become NaN (genuine missingness, which the
    mask must see), while values merely outside [valid_low, valid_high] are
    winsorized to the nearest valid bound.
    """
    out = df_vitals.copy()
    for col in out.columns:
        parts = col.split("_")
        vital = parts[1] if len(parts) > 1 else None
        if vital not in VITALS_VALID_RANGE:
            continue
        r = VITALS_VALID_RANGE[vital]
        s = out[col].astype(float)
        if vital == "temperature":
            s = (s - 32.0) * 5.0 / 9.0  # convert_temp_to_celcius()
        s = s.mask((s < r["outlier_low"]) | (s > r["outlier_high"]), np.nan)
        s = s.clip(lower=r["valid_low"], upper=r["valid_high"])
        out[col] = s
    return out


def _load_master_vitals_lookup():
    """Loads data/master_dataset.csv's PRE-imputation vitals (stay_id-
    indexed) once per process, then applies _clean_vitals_frame().

    See M2_Hybrid_Deviation_Log.md Deviation 4: data_general_processing.py
    median-imputes all 14 vitals columns before train.csv/test.csv are ever
    written (confirmed: 0 nulls, 100% of rows, in the current train.csv),
    so those files cannot be used to recover real missingness --
    master_dataset.csv, the pre-imputation stage, still has it. Deviation 7
    adds the cleaning step that switching source had dropped. stay_id is a
    verified unique key present for every train/test row, so this is a safe
    1:1 lookup, not an approximation."""
    global _MASTER_VITALS_LOOKUP
    if _MASTER_VITALS_LOOKUP is None:
        _assert_valid_range_matches_pipeline()
        triage_vitals = [f"triage_{c}" for c in VITALS_COLS]
        ed_vitals = [f"ed_{c}_last" for c in VITALS_COLS]
        master = pd.read_csv(
            os.path.join(DATA_DIR, "master_dataset.csv"),
            usecols=["stay_id"] + triage_vitals + ed_vitals,
        )
        assert not master["stay_id"].duplicated().any(), "master_dataset.csv stay_id is not unique"
        master = master.set_index("stay_id")
        _MASTER_VITALS_LOOKUP = _clean_vitals_frame(master)
    return _MASTER_VITALS_LOOKUP


def get_vitals_scaler():
    """Per-vital mean/SD (plus a Delta-t reference scale) for the temporal
    models, fit on the FIXED `training` subject subset ONLY -- never
    validation, never either calibration subset, never test.

    Fitting this is leak-free and seed-independent by construction: the
    4-way development-pool partition (build_fixed_dev_split) is drawn once
    at random_state=42 and persisted, so the `training` subject set is a
    constant of this experiment, not a per-seed or per-run quantity. That
    is why the scaler can live behind a module-level cache and be applied
    inside prepare_temporal_tensors() without every call site threading it
    through -- every script gets identical, leak-free scaling with no way
    to forget the fit-on-train-only rule.

    Why scaling is needed at all (M2_Hybrid_Deviation_Log.md Deviation 7):
    the temporal path previously fed RAW vitals into a tanh/sigmoid GRU --
    heart rate ~85, SBP ~135, temperature ~98 (Fahrenheit, pre-cleaning).
    Saturated gates at those magnitudes are the direct cause of the
    training pathologies previously recorded as protocol deviations
    (Deviation 3's "lr=0.01+batch_size=128 never learns", Deviation 5's
    2000-epoch budget and post-epoch-2100 instability).

    Statistics are computed with NaNs excluded, so real missingness never
    biases the mean/SD. Missing entries are encoded as 0.0 AFTER scaling,
    which in scaled space is exactly the training-set empirical mean --
    i.e. GRU-D's own x_tilde term (Che et al. 2018), not an arbitrary
    sentinel. Cached to data/m2_vitals_scaler.json for inspection.
    """
    global _VITALS_SCALER
    if _VITALS_SCALER is not None:
        return _VITALS_SCALER
    if os.path.exists(_SCALER_PATH):
        with open(_SCALER_PATH) as f:
            cached = json.load(f)
        if cached.get("version") == SCALER_VERSION:
            _VITALS_SCALER = cached
            return _VITALS_SCALER

    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                     usecols=["subject_id", "stay_id", "ed_los_hours"])
    df = df[df["subject_id"].isin(train_ids)]
    lookup = _load_master_vitals_lookup()
    vitals = lookup.loc[df["stay_id"].values]

    mean, sd = {}, {}
    for c in VITALS_COLS:
        # One scaler per VITAL, not per column: triage_x and ed_x_last are
        # the same physiological quantity at two timesteps and must share a
        # scale, or the GRU would see an artificial step change between
        # t=0 and t=1 for every patient.
        pooled = np.concatenate([vitals[f"triage_{c}"].values, vitals[f"ed_{c}_last"].values])
        mean[c] = float(np.nanmean(pooled))
        s = float(np.nanstd(pooled))
        sd[c] = s if s > 1e-8 else 1.0

    los = df["ed_los_hours"].astype(float).clip(lower=0.0)
    delta_ref = float(np.nanmedian(los))
    _VITALS_SCALER = {
        "version": SCALER_VERSION,
        "mean": mean, "sd": sd,
        "delta_ref_hours": delta_ref if delta_ref > 1e-8 else 1.0,
        "n_train_rows": int(len(df)), "n_train_patients": int(len(train_ids)),
        "fit_on": "fixed dev-pool `training` subject subset only",
    }
    with open(_SCALER_PATH, "w") as f:
        json.dump(_VITALS_SCALER, f, indent=2)
    return _VITALS_SCALER


def prepare_temporal_tensors(df):
    """Returns (X, M, D, y): the 2-timestep (triage, latest ED) x 7-vitals
    GRU-D input as float32 numpy arrays, shape (N,2,7)/(N,2,7)/(N,2,7)/(N,).

    X is standardized per vital using get_vitals_scaler() (fit on the fixed
    `training` subset only); missing entries are 0.0 in that scaled space,
    which equals the training empirical mean and is exactly the x_tilde
    term GRU-D's decay interpolates toward. M is the genuine observation
    mask. D is Delta-t in units of the training-set median ED length of
    stay: d=0 at triage, d=ed_los_hours/delta_ref at the ED reading.

    History (M2_Hybrid_Deviation_Log.md): Deviation 4 moved the vitals
    source to master_dataset.csv so M stops being all-ones. Deviation 7
    adds back the Celsius conversion + outlier removal that move had
    silently dropped, adds train-only standardization, and clips the 6
    negative ed_los_hours values in the cohort to 0 (a negative elapsed
    time is not a meaningful Delta-t).
    """
    triage_vitals = [f"triage_{c}" for c in VITALS_COLS]
    ed_vitals = [f"ed_{c}_last" for c in VITALS_COLS]
    lookup = _load_master_vitals_lookup()
    real_vitals = lookup.loc[df["stay_id"].values, triage_vitals + ed_vitals]
    sc = get_vitals_scaler()
    mu = np.array([sc["mean"][c] for c in VITALS_COLS], dtype=np.float64)
    sd = np.array([sc["sd"][c] for c in VITALS_COLS], dtype=np.float64)

    T_raw = real_vitals[triage_vitals].values.astype(np.float64)
    E_raw = real_vitals[ed_vitals].values.astype(np.float64)
    T_m = (~np.isnan(T_raw)).astype(np.float64)
    E_m = (~np.isnan(E_raw)).astype(np.float64)
    # Standardize, then encode missing as 0.0 (== the training mean).
    T_v = np.nan_to_num((T_raw - mu) / sd, nan=0.0)
    E_v = np.nan_to_num((E_raw - mu) / sd, nan=0.0)

    if "ed_los_hours" in df.columns:
        los = df["ed_los_hours"].astype(float)
        ed_los = los.fillna(los.median()).clip(lower=0.0).values
    else:
        ed_los = np.zeros(len(df))
    ed_los = ed_los / sc["delta_ref_hours"]

    X = np.stack([T_v, E_v], axis=1).astype(np.float32)
    M = np.stack([T_m, E_m], axis=1).astype(np.float32)
    d0 = np.zeros_like(T_m)
    d1 = np.tile(ed_los[:, None], (1, len(VITALS_COLS)))
    D = np.stack([d0, d1], axis=1).astype(np.float32)
    y = df[TARGET].astype(int).values
    return X, M, D, y


def _to_tensor(a):
    return torch.tensor(a, dtype=torch.float32, device=DEVICE)


# GRUDCell lives in its own module so m2_ablations.py (Track A) can share
# the exact same recurrence without a circular import -- see m2_grud_cell.py.
# Re-exported here so existing callers of m2_hybrid_common keep working.
from m2_grud_cell import GRUDCell, GRUD_CELL_PARAM_KEYS  # noqa: F401,E402


class GRUDWithEncoder(nn.Module):
    """GRUDCell + a 2-layer classifier head. encode() exposes the raw
    hidden state h BEFORE the head, which Guide v3.8 Sec 2.1 needs in order
    to freeze just the encoder and discard the head entirely ("the temporal
    prediction head is discarded and not used in the Hybrid"). Stage 1
    (Temporal-Only GRU-D) trains this directly with stat=0, satisfying Sec
    2.1's "reuse the existing GRU-D implementation, but set static_size=0"
    literally."""

    def __init__(self, inp=7, hid=32, stat=0, out=1, clf_hidden=32, dropout=0.2):
        super().__init__()
        self.hid = hid
        self.cell = GRUDCell(inp=inp, hid=hid)
        self.clf = nn.Sequential(
            nn.Linear(hid + stat, clf_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(clf_hidden, out),
        )

    def encode(self, X, M, D):
        return self.cell(X, M, D)

    def forward(self, X, M, D, S=None):
        h = self.encode(X, M, D)
        if S is not None:
            h = torch.cat([h, S], 1)
        return self.clf(h)


class TemporalEncoder(nn.Module):
    """The frozen GRU-D recurrent encoder alone -- no classifier head.
    Loads its parameters from a Stage-1-trained GRUDWithEncoder checkpoint
    via load_state_dict(..., strict=False), so the checkpoint's
    now-discarded clf.* keys are ignored rather than erroring; the
    `cell.*` keys line up by construction because both classes hold the
    same GRUDCell submodule under the same attribute name.
    load_frozen_encoder() below asserts every expected `cell.*` key was
    actually found, so strict=False can never silently leave a
    randomly-initialised encoder in place. This is the object Phases J/K
    (Frozen-Temporal Head Control, Hybrid GRU-D) load per seed and never
    update -- see Guide Sec 2.1's "Critical freeze rule"."""

    def __init__(self, inp=7, hid=32):
        super().__init__()
        self.hid = hid
        self.cell = GRUDCell(inp=inp, hid=hid)

    def forward(self, X, M, D):
        return self.cell(X, M, D)


def load_temporal_only_model(seed, hidden=32, dropout=0.2):
    """Loads Stage 1's full (encoder+head) checkpoint for evaluating
    Temporal-Only GRU-D's own predictions (conformal analysis, unified
    predictions file) -- distinct from load_frozen_encoder() below, which
    loads only the encoder (discarding the head) for Phases J/K."""
    model = GRUDWithEncoder(inp=7, hid=hidden, stat=0, clf_hidden=32, dropout=dropout).to(DEVICE)
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"m2_temporal_only_grud_seed{seed}.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_fusion_head(checkpoint_prefix, seed, temporal_dim=32, static_hidden=None,
                      static_dropout=0.2, fusion_dropout=0.1):
    """Loads a saved FusionHead checkpoint (Frozen-Temporal Head Control's
    or Hybrid GRU-D's head) for a given seed. Callers should read the
    static_hidden/static_dropout/fusion_dropout values from that model's
    own saved per-seed JSON config rather than hardcoding them, since the
    architecture must match exactly for load_state_dict to succeed."""
    static_encoder = (StaticEncoder(in_dim=4, hidden=static_hidden, dropout=static_dropout).to(DEVICE)
                       if static_hidden is not None else None)
    head = FusionHead(temporal_dim=temporal_dim, static_encoder=static_encoder,
                       fusion_dropout=fusion_dropout).to(DEVICE)
    ckpt = torch.load(os.path.join(CHECKPOINT_DIR, f"{checkpoint_prefix}_seed{seed}.pt"), map_location=DEVICE)
    head.load_state_dict(ckpt)
    head.eval()
    return head


def load_frozen_encoder(seed, hid=32):
    """Loads seed's frozen encoder from the Stage-1 checkpoint saved by
    m2_temporal_only_grud.py, freezes its parameters (requires_grad=False),
    and sets eval mode. Never shared across seeds (Guide Sec 7: "No
    cross-seed encoder sharing")."""
    encoder = TemporalEncoder(inp=7, hid=hid).to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"m2_temporal_only_grud_seed{seed}.pt")
    state = torch.load(ckpt_path, map_location=DEVICE)
    missing, _unexpected = encoder.load_state_dict(state, strict=False)
    # strict=False is required (the checkpoint carries the discarded clf.*
    # head) but on its own it would also silently tolerate a checkpoint
    # whose encoder keys do not match at all, leaving a randomly-initialised
    # "frozen encoder" behind and quietly invalidating every downstream
    # result. Assert the encoder half loaded completely.
    assert not missing, (
        f"Frozen encoder checkpoint {ckpt_path} is missing {missing} -- it was almost "
        f"certainly written by the pre-Deviation-7 architecture. Re-run "
        f"m2_temporal_only_grud.py before any downstream stage. STOP."
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


class StaticEncoder(nn.Module):
    """Guide Sec 2.2: the 4-static-feature branch. One hidden layer
    (deliberately simple, per Sec 2.2/Sec 6.2), ReLU, dropout, 32-d
    output embedding -- no output activation (the fusion head that
    consumes this embedding applies its own final sigmoid)."""

    def __init__(self, in_dim=4, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, s):
        return self.net(s)


class FusionHead(nn.Module):
    """Guide Sec 2.3/Sec 4.2: with static_encoder=None, this IS the
    Frozen-Temporal Head Control's head (Linear(32,1) after a dropout);
    with a StaticEncoder instance, this is Hybrid's fusion head
    (concatenate 32-d temporal + 32-d static = 64-d, Linear(64,1)). Using
    the identical class for both means the two models' heads are trained
    by verifiably the same code, not just similarly-shaped code -- any
    Hybrid-vs-control difference is then attributable only to the
    presence of the static branch itself."""

    def __init__(self, temporal_dim=32, static_encoder=None, fusion_dropout=0.1):
        super().__init__()
        self.static_encoder = static_encoder
        static_dim = static_encoder.net[0].out_features if static_encoder is not None else 0
        self.dropout = nn.Dropout(fusion_dropout)
        self.fc = nn.Linear(temporal_dim + static_dim, 1)

    def forward(self, h_temporal, s=None):
        if self.static_encoder is not None:
            h = torch.cat([h_temporal, self.static_encoder(s)], dim=1)
        else:
            h = h_temporal
        return self.fc(self.dropout(h))


def _minibatch_early_stop_loop(param_groups_fn, forward_fn, X_fit_tensors, y_fit, X_val_tensors, y_val,
                                seed, lr, weight_decay, batch_size, max_epochs, patience, train_modules):
    """Shared mini-batch + validation-loss-early-stopping training loop
    used by both train_temporal_only() and train_on_frozen_encoder() below
    -- factored out so the two training procedures are guaranteed to
    early-stop, shuffle, and select the best epoch identically, per Guide
    Sec 4.2's requirement that Frozen-Temporal Head Control and Hybrid use
    verifiably the same training protocol."""
    opt = optim.Adam(param_groups_fn(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    yt = _to_tensor(y_fit).unsqueeze(1)
    yv = _to_tensor(y_val).unsqueeze(1)
    n = yt.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best_val_loss, patience_left, best_states = float("inf"), patience, None
    history = []
    for epoch in range(max_epochs):
        for m in train_modules:
            m.train()
        if batch_size is None:
            # Full-batch: one gradient step per epoch, matching
            # SKLearnGRUD's own batch_size=None convention in
            # m2_ablations.py. See M2_Hybrid_Deviation_Log.md Deviation 4
            # for why this is used instead of Guide Sec 6.1's literal
            # batch_size=128 for Temporal-Only GRU-D.
            opt.zero_grad()
            logits = forward_fn(X_fit_tensors)
            loss = crit(logits, yt)
            loss.backward()
            opt.step()
        else:
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                logits = forward_fn([t[idx] for t in X_fit_tensors])
                loss = crit(logits, yt[idx])
                loss.backward()
                opt.step()

        for m in train_modules:
            m.eval()
        with torch.no_grad():
            val_logits = forward_fn(X_val_tensors)
            val_loss = crit(val_logits, yv).item()
            val_auroc = roc_auc_score(y_val, torch.sigmoid(val_logits).cpu().numpy().ravel())
        history.append({"epoch": epoch, "val_loss": val_loss, "val_auroc": val_auroc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = patience
            best_states = [{k: v.clone() for k, v in m.state_dict().items()} for m in train_modules]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    for m, state in zip(train_modules, best_states):
        m.load_state_dict(state)
    return history


def train_temporal_only(X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed,
                         hidden=32, dropout=0.2, weight_decay=1e-4, lr=0.01,
                         batch_size=128, max_epochs=100, patience=10):
    """Stage 1 (Guide Sec 2.1/6.1): jointly trains a GRUDWithEncoder
    (stat=0) end-to-end via mini-batch Adam, early-stopped on VALIDATION
    LOSS (patience=10) -- a deliberate difference from the pre-v3.8
    pipeline's AUROC-based early stopping, since Sec 6.1 gives an explicit
    instruction for this new component. Returns (model, history); the
    caller is responsible for verifying (history) that max_epochs was
    actually sufficient for convergence before trusting any result --
    see M2_Findings_Dossier.md Sec 12's Static-Only MLP epoch-budget bug
    for why this check is mandatory, not optional, given batch_size=128
    is a materially different training regime from anything tried before.
    """
    torch.manual_seed(seed)
    model = GRUDWithEncoder(inp=7, hid=hidden, stat=0, clf_hidden=32, dropout=dropout).to(DEVICE)
    X_fit_t = [_to_tensor(X_fit), _to_tensor(M_fit), _to_tensor(D_fit)]
    X_val_t = [_to_tensor(X_val), _to_tensor(M_val), _to_tensor(D_val)]

    history = _minibatch_early_stop_loop(
        param_groups_fn=lambda: model.parameters(),
        forward_fn=lambda ts: model(ts[0], ts[1], ts[2]),
        X_fit_tensors=X_fit_t, y_fit=y_fit, X_val_tensors=X_val_t, y_val=y_val,
        seed=seed, lr=lr, weight_decay=weight_decay, batch_size=batch_size,
        max_epochs=max_epochs, patience=patience, train_modules=[model],
    )
    return model, history


def train_on_frozen_encoder(encoder, X_fit, M_fit, D_fit, y_fit, X_val, M_val, D_val, y_val, seed,
                             S_fit=None, S_val=None, static_hidden=None, static_dropout=0.2,
                             fusion_dropout=0.1, lr=1e-3, weight_decay=0.0,
                             batch_size=128, max_epochs=100, patience=10):
    """Stages 2/3 (Guide Sec 4.2/2.3/6.5): trains a fresh head -- with no
    static branch (Frozen-Temporal Head Control, static_hidden=None) or a
    fresh static-MLP-encoder + fusion head (Hybrid, static_hidden given)
    -- on top of a FROZEN, already-trained TemporalEncoder. The encoder's
    parameters never receive gradients (Guide's gradient constraint: only
    static-branch and fusion-head parameters are optimized). The frozen
    encoder's embeddings are precomputed once (it's in eval mode and
    produces the same h for the same input every call, so there's no need
    to re-run the GRU loop every epoch or every tuning config). Returns
    (head, history) -- head.static_encoder is None or the trained
    StaticEncoder submodule.
    """
    encoder.eval()
    with torch.no_grad():
        h_fit = encoder(_to_tensor(X_fit), _to_tensor(M_fit), _to_tensor(D_fit)).detach()
        h_val = encoder(_to_tensor(X_val), _to_tensor(M_val), _to_tensor(D_val)).detach()

    static_encoder = None
    s_fit_t = s_val_t = None
    if static_hidden is not None:
        torch.manual_seed(seed)
        static_encoder = StaticEncoder(in_dim=S_fit.shape[1], hidden=static_hidden, dropout=static_dropout).to(DEVICE)
        s_fit_t = _to_tensor(S_fit)
        s_val_t = _to_tensor(S_val)

    head = FusionHead(temporal_dim=h_fit.shape[1], static_encoder=static_encoder,
                       fusion_dropout=fusion_dropout).to(DEVICE)

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


def predict_proba_temporal_only(model, X, M, D):
    model.eval()
    with torch.no_grad():
        logits = model(_to_tensor(X), _to_tensor(M), _to_tensor(D))
        return torch.sigmoid(logits).cpu().numpy().ravel()


def predict_proba_frozen(encoder, head, X, M, D, S=None):
    encoder.eval()
    head.eval()
    with torch.no_grad():
        h = encoder(_to_tensor(X), _to_tensor(M), _to_tensor(D))
        s_t = _to_tensor(S) if S is not None else None
        logits = head(h, s_t)
        return torch.sigmoid(logits).cpu().numpy().ravel()


# ── Conformal layer for the 3 GRU-D-family models (Guide v3.8 Sec 2.4) ──────
# Applies to Temporal-Only GRU-D, Frozen-Temporal Head Control, and Hybrid
# GRU-D only (Static-Only MLP/XGBoost-Matched stay discrimination+
# calibration-only, unchanged from Phases C/D).
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402


class PrecomputedProbaClassifier(BaseEstimator, ClassifierMixin):
    """Adapter exposing a precomputed array of P(y=1) as an sklearn-style
    classifier's predict_proba, so MAPIE's SplitConformalClassifier can
    conformalize/predict_set against models (our GRU-D-family
    encoder+head pairs) that don't take a plain 2D feature matrix as
    input. `X` fed to this adapter is a trivial row-index array; the real
    model forward pass happens once, outside of MAPIE's call cycle, via
    predict_proba_frozen()/predict_proba_temporal_only()."""

    def __init__(self, y_prob):
        self.y_prob = np.asarray(y_prob)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        idx = np.asarray(X).ravel().astype(int)
        p1 = self.y_prob[idx]
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        idx = np.asarray(X).ravel().astype(int)
        return (self.y_prob[idx] > 0.5).astype(int)


def run_mapie_classification(y_prob_final_cal, y_final_cal, y_prob_test):
    """Guide Sec 2.4: 'MapieClassifier with method="score"' maps onto
    mapie 1.5.0's SplitConformalClassifier(conformity_score='lac') --
    verified directly against the installed mapie's own source
    (LACConformityScore's docstring: "also formerly called `score`",
    citing Sadinle, Lei & Wasserman, "Least Ambiguous Set-Valued
    Classifiers with Bounded Error Levels," JASA 114, 2019). Calibrated on
    the model's ALREADY-COMPUTED probabilities for the
    `final_conformal_calibration` subject subset (Sec 2.4's 4-way split --
    never the calibration_selection subset used for ECI's lr sweep, nor
    the test set). Both probability arrays are concatenated into one
    lookup table with non-overlapping index ranges for calibration vs.
    test, so the adapter's public sklearn interface (no private mapie
    attribute access) can serve both calls correctly.

    Returns a (n_test, 2) boolean prediction-set matrix (mapie's own
    predict_set() shape verified directly: (n,2,1) for a single
    confidence_level, squeezed here to (n,2)); coverage = fraction where
    the true class's slot is True, mean set size = mean row-sum.
    """
    from mapie.classification import SplitConformalClassifier
    y_prob_final_cal = np.asarray(y_prob_final_cal)
    y_prob_test = np.asarray(y_prob_test)
    combined = np.concatenate([y_prob_final_cal, y_prob_test])
    n_cal = len(y_prob_final_cal)

    clf = PrecomputedProbaClassifier(combined)
    scc = SplitConformalClassifier(estimator=clf, confidence_level=0.9,
                                    conformity_score="lac", prefit=True)
    idx_cal = np.arange(0, n_cal).reshape(-1, 1)
    scc.conformalize(idx_cal, np.asarray(y_final_cal))
    idx_test = np.arange(n_cal, n_cal + len(y_prob_test)).reshape(-1, 1)
    _, y_sets = scc.predict_set(idx_test)
    return y_sets[:, :, 0]


def run_eci_4way(y_prob_cal_sel, y_true_cal_sel, y_prob_final_cal, y_true_final_cal,
                  y_prob_test, y_true_test):
    """Guide Sec 2.4's two-stage ECI selection, replacing the pre-v3.8
    single-fold approach:
      Stage 1: sweep lr in {0.01,0.05,0.1} on `calibration_selection`
        ALONE (a self-contained online run: ECI warms up and is evaluated
        within the same subset) -- select by coverage closest to 90%,
        narrowest width as tiebreak (same rule as the original v3.2 ECI-lr
        sweep). Freeze the winning lr.
      Stage 2: apply ECI at the frozen lr as ONE continuous online stream
        over final_conformal_calibration (warm-up prefix) + test
        (evaluated suffix) -- final_conformal_calibration establishes q
        before test is ever reached, matching Sec 2.4's "threshold...
        determined by the final conformal-calibration subset after ECI-LR
        selection is frozen," while respecting ECI's own online design
        (it is not a fit-once-apply-later method the way MAPIE is).
    Reuses eci_method/_sorted_bounds from m2_ablations.py, same
    |y_true-y_prob| nonconformity score convention as run_eci().

    Returns (selected_lr, coverage_test, width_test, lo_test, hi_test,
    stage1_candidates).
    """
    from m2_ablations import eci_method  # already sys.path-wired there (core.methods.ECI)
    y_prob_cal_sel = np.asarray(y_prob_cal_sel)
    y_true_cal_sel = np.asarray(y_true_cal_sel, dtype=float)

    candidates = []
    for lr in (0.01, 0.05, 0.1):
        scores = np.abs(y_true_cal_sel - y_prob_cal_sel)
        res = eci_method(scores=scores, alpha=0.1, lr=lr, T_burnin=5, ahead=1, proportional_lr=True)
        qs = res["q"]
        lo, hi = _sorted_bounds(y_prob_cal_sel - qs, y_prob_cal_sel + qs)
        lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
        cov = float(np.mean((y_true_cal_sel >= lo) & (y_true_cal_sel <= hi)))
        wid = float(np.mean(hi - lo))
        candidates.append({"lr": lr, "coverage": cov, "width": wid, "coverage_gap": abs(cov - 0.9)})
    candidates.sort(key=lambda c: (c["coverage_gap"], c["width"]))
    selected_lr = candidates[0]["lr"]

    y_true_stream = np.concatenate([np.asarray(y_true_final_cal, dtype=float), np.asarray(y_true_test, dtype=float)])
    y_prob_stream = np.concatenate([np.asarray(y_prob_final_cal), np.asarray(y_prob_test)])
    scores_stream = np.abs(y_true_stream - y_prob_stream)
    res = eci_method(scores=scores_stream, alpha=0.1, lr=selected_lr, T_burnin=5, ahead=1, proportional_lr=True)
    qs = res["q"]
    lo, hi = _sorted_bounds(y_prob_stream - qs, y_prob_stream + qs)
    lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)

    n_cal = len(y_true_final_cal)
    lo_test, hi_test = lo[n_cal:], hi[n_cal:]
    y_true_test_arr = np.asarray(y_true_test, dtype=float)
    cov_test = float(np.mean((y_true_test_arr >= lo_test) & (y_true_test_arr <= hi_test)))
    wid_test = float(np.mean(hi_test - lo_test))
    return selected_lr, cov_test, wid_test, lo_test, hi_test, candidates


# ── Probabilistic calibration metrics (Guide Sec 8) ─────────────────────────
# Required for every model (discrimination is just roc_auc_score/
# average_precision_score, already used throughout the existing pipeline).
# Brier score is sklearn's own brier_score_loss; ECE has no sklearn builtin,
# so it's implemented here once, shared by every new M2 script, using the
# standard equal-width-binned definition (mean |accuracy-confidence| gap,
# weighted by bin occupancy).
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def conformal_coverage_report(y_true, covered, set_size=None, prefix=""):
    """Marginal AND class-conditional coverage for a conformal procedure.

    Why class-conditional coverage is reported alongside the marginal
    number, and why it is not optional here: split-conformal and ECI both
    guarantee only MARGINAL coverage, and this cohort is 94.1% negative.
    Measured directly on the pre-Deviation-7 Hybrid GRU-D at seed 42, the
    marginal numbers looked healthy while the clinically relevant class did
    not:

        MAPIE : marginal 0.9021 | class 0 = 0.9545 | class 1 = 0.0629
        ECI   : marginal 0.9218 | class 0 = 0.9722 | class 1 = 0.1150

    i.e. the 90% target was being met almost entirely by covering the
    negative majority, while coverage for patients who actually
    deteriorated was 6-12%. A marginal-only table cannot show that, so
    every conformal result in this module now carries the per-class
    breakdown next to it.

    `set_size` (MAPIE classification only) additionally yields the
    empty-set rate. A mean set size below 1.0 in a binary task is
    achievable ONLY via empty prediction sets -- "no label is plausible at
    this confidence" -- so a sub-1.0 mean must never be reported as a
    tightness result without the empty-set rate beside it.
    """
    y_true = np.asarray(y_true).astype(int)
    covered = np.asarray(covered).astype(bool)
    out = {f"{prefix}coverage": float(covered.mean())}
    for c in (0, 1):
        m = y_true == c
        out[f"{prefix}coverage_class{c}"] = float(covered[m].mean()) if m.any() else float("nan")
        out[f"{prefix}n_class{c}"] = int(m.sum())
    if set_size is not None:
        set_size = np.asarray(set_size)
        out[f"{prefix}empty_set_rate"] = float((set_size == 0).mean())
        for c in (0, 1):
            m = y_true == c
            out[f"{prefix}empty_set_rate_class{c}"] = (
                float((set_size[m] == 0).mean()) if m.any() else float("nan"))
    return out


def calibration_slope_intercept(y_true, y_prob, eps=1e-6):
    """Cox calibration regression: fit y ~ logit(p_hat) via unpenalized
    logistic regression; the fitted slope/intercept summarize over/under-
    confidence (slope<1 = predictions too extreme, intercept!=0 = systematic
    over/under-estimation). Returns (slope, intercept), or (nan, nan) if the
    fit is degenerate (e.g. a single class in y_true)."""
    from sklearn.linear_model import LogisticRegression
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(penalty=None, max_iter=1000)
    lr.fit(logit_p, y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

# ── The 4 Guide-mandated static features (Sec 3) ────────────────────────────
HYBRID_STATIC_COLS = ["age", "sex", "triage_acuity", "comorbidity_elixhauser_count"]


def add_hybrid_static_features(df):
    """Returns a copy of df with the 4 Hybrid-Guide static features added.

    - age, triage_acuity: reused as-is (already present, already validated
      non-null in the filtered cohort per M2_Outcome_Definition_Gate.md).
    - sex: gender re-encoded M=1/F=0 (same convention as static_cols in
      m2_ablations.py, for consistency across M2 scripts).
    - comorbidity_elixhauser_count: row-wise sum of the 31 real eci_* flags
      already populated by the vendored commorbidity() extraction step
      (medcode_utils.py, citing Quan H, Sundararajan V, Halfon P, et al.,
      "Coding algorithms for defining Comorbidities in ICD-9-CM and ICD-10
      administrative data," Med Care, 2005 Nov;43(11):1130-9 -- verified
      directly against Benchmark_scripts/medcodes/diagnoses/comorbidities.py's
      own docstring, not asserted from memory). This is 31 categories, not
      the 29 the Hybrid Guide states -- see the Clarification Request's FYI
      section. A full-file dtype check of train.csv/test.csv found all 31
      columns are clean int64 (no non-numeric artifacts); pd.to_numeric
      coercion below is applied defensively regardless, since this feeds a
      frozen comparison table.
    """
    df = df.copy()
    df["sex"] = df["gender"].map({"M": 1, "F": 0, "m": 1, "f": 0}).fillna(0).astype(int)
    eci_cols = [c for c in df.columns if c.startswith("eci_")]
    assert len(eci_cols) == 31, f"expected 31 Elixhauser flags, found {len(eci_cols)}"
    df["comorbidity_elixhauser_count"] = (
        df[eci_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    )
    return df


_STATIC_SCALER = None
_STATIC_SCALER_PATH = os.path.join(DATA_DIR, "m2_static_scaler.json")


# -- Real-calendar temporal axis (MIMIC-IV date de-shifting) ---------------
MIMIC4_DIR = os.path.join(DATA_DIR, "..", "mimic4_full")
_REAL_TIME_LOOKUP = None


def _load_real_time_lookup():
    """stay_id -> approximate REAL calendar year of the ED visit.

    Why raw `intime` cannot be used for this. MIMIC-IV de-identifies dates
    by shifting every patient's timeline by a random per-patient offset
    into the future: `anchor_year` in this cohort's `patients` table ranges
    2110-2208, and `intime` values sit in the same shifted space (verified
    directly: 2138, 2181, 2183, 2192 all appear in test.csv). Intra-patient
    intervals are preserved, but the shift is independent per patient, so
    comparing `intime` ACROSS patients carries no real-time meaning at all.
    Sorting a multi-patient stream by `intime` therefore produces an
    essentially arbitrary order wearing a chronological disguise.

    What is recoverable is `anchor_year_group` -- a real 3-year window
    (2008-2010 ... 2020-2022) that the patient's own shifted `anchor_year`
    corresponds to. The standard MIMIC-IV recipe for approximating the real
    time of an individual event is therefore:

        real_year ~= start(anchor_year_group) + (event_year - anchor_year)

    Resolution and its limits, stated plainly because they bound every
    conclusion drawn from this axis: `anchor_year_group` is a 3-year
    bucket, so a recovered year carries roughly +/-1 year of uncertainty,
    and encounters landing exactly on a patient's anchor year pile up on
    the bucket's start year (visible as spikes at 2011/2014/2017 in the
    cohort's year histogram). The axis is trustworthy for coarse
    past-vs-future ordering and for a multi-year train/test cut; it is NOT
    a precise arrival-time ordering, and no within-year cross-patient order
    is recoverable at all.
    """
    global _REAL_TIME_LOOKUP
    if _REAL_TIME_LOOKUP is not None:
        return _REAL_TIME_LOOKUP
    patients = pd.read_csv(os.path.join(MIMIC4_DIR, "hosp", "patients.csv"),
                            usecols=["subject_id", "anchor_year", "anchor_year_group"])
    patients["grp_start"] = patients["anchor_year_group"].str.slice(0, 4).astype(int)
    master = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"),
                          usecols=["stay_id", "subject_id", "intime"])
    m = master.merge(patients, on="subject_id", how="left")
    # master_dataset.csv spans the WHOLE extraction, including encounters the
    # Gate's cohort filters later remove (age < 18, null triage acuity). A
    # handful of those patients are absent from patients.csv (76 master rows
    # / 39 patients, verified) and simply get a NaN year here. That is not an
    # error at this level -- the strict check belongs on the rows a caller
    # actually asks for, which add_temporal_axis() enforces. Verified
    # directly: 0 of those rows fall inside the frozen train/test cohort.
    ev_year = pd.to_datetime(m["intime"]).dt.year
    m["approx_real_year"] = m["grp_start"] + (ev_year - m["anchor_year"])
    _REAL_TIME_LOOKUP = m.set_index("stay_id")[["approx_real_year"]]
    return _REAL_TIME_LOOKUP


def add_temporal_axis(df):
    """Adds `approx_real_year` to df (joined on stay_id)."""
    lookup = _load_real_time_lookup()
    out = df.copy()
    out["approx_real_year"] = lookup.loc[out["stay_id"].values, "approx_real_year"].values
    assert out["approx_real_year"].notna().all(), "stay_id missing from the real-time lookup -- STOP."
    return out


def sort_chronologically(df):
    """Orders encounters by approximate real calendar year, with `intime`
    as a deterministic (but NOT real-time-meaningful) tiebreak inside a
    year -- see _load_real_time_lookup() for why no finer real ordering
    exists. Use this, never `sort_values("intime")`, wherever a stream is
    meant to arrive in time order."""
    out = add_temporal_axis(df)
    by = ["approx_real_year", "intime"] if "intime" in out.columns else ["approx_real_year"]
    return out.sort_values(by).reset_index(drop=True)


def get_static_scaler():
    """Mean/SD for HYBRID_STATIC_COLS, fit on the fixed `training` subject
    subset ONLY -- the same subset, and therefore numerically the same
    transform, that m2_static_only_mlp.py already fits its own
    StandardScaler on.

    Why this exists (M2_Hybrid_Deviation_Log.md Deviation 7): Static-Only
    MLP standardised these 4 columns, but Hybrid GRU-D's static branch
    consumed them RAW -- age in years (18-103), triage_acuity (1-5), sex
    (0/1) and an Elixhauser count (0-31) fed straight into a
    ReLU/dropout MLP whose output is concatenated with a 32-d temporal
    embedding. Two of the five headline deltas
    (Delta_static_contribution, Delta_static_only) compare exactly those
    two models, so the comparison was confounded by preprocessing rather
    than by the presence of the static branch. Sharing one scaler makes
    the treatment verifiably identical instead of merely similar.
    """
    global _STATIC_SCALER
    if _STATIC_SCALER is not None:
        return _STATIC_SCALER
    if os.path.exists(_STATIC_SCALER_PATH):
        with open(_STATIC_SCALER_PATH) as f:
            _STATIC_SCALER = json.load(f)
        return _STATIC_SCALER

    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df = df[df["subject_id"].isin(train_ids)]
    df = add_hybrid_static_features(df)
    vals = df[HYBRID_STATIC_COLS].values.astype(np.float64)
    mean = np.nanmean(vals, axis=0)
    sd = np.nanstd(vals, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    _STATIC_SCALER = {
        "cols": list(HYBRID_STATIC_COLS),
        "mean": [float(v) for v in mean],
        "sd": [float(v) for v in sd],
        "n_train_rows": int(len(df)),
        "fit_on": "fixed dev-pool `training` subject subset only",
    }
    with open(_STATIC_SCALER_PATH, "w") as f:
        json.dump(_STATIC_SCALER, f, indent=2)
    return _STATIC_SCALER


def prepare_static_features(df):
    """The 4 Guide static features, standardised with get_static_scaler().
    `df` must already have passed through add_hybrid_static_features().
    Every model with a neural static branch must go through this function
    -- never `df[HYBRID_STATIC_COLS].values` directly."""
    sc = get_static_scaler()
    assert sc["cols"] == list(HYBRID_STATIC_COLS), "static scaler column order drift -- STOP."
    vals = df[HYBRID_STATIC_COLS].values.astype(np.float64)
    mean = np.asarray(sc["mean"])
    sd = np.asarray(sc["sd"])
    return np.nan_to_num((vals - mean) / sd, nan=0.0).astype(np.float32)


def load_fixed_test_subject_ids():
    """The single source of truth for the fixed M2 test cohort. Never
    re-derive the split -- reuse data/test.csv's own subject_id column,
    exactly as the Guide Sec 5.1 requires ("Reuse the exact same test
    subject_ids as the frozen GRU-D run")."""
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), usecols=["subject_id"])
    return np.sort(df_test["subject_id"].unique())


def _hash_ids(ids):
    return hashlib.sha256(",".join(str(i) for i in np.sort(np.asarray(ids)).tolist()).encode()).hexdigest()


def _split_ids(ids, fracs, seed):
    """Sequentially partitions a 1-D array of unique IDs into len(fracs)
    disjoint, exhaustive groups matching fracs (must sum to 1.0). Chains
    sklearn's train_test_split at one fixed random_state, matching
    patient_level_split's own convention elsewhere in this pipeline, but
    operating on bare subject IDs (no accompanying X/y rows needed here)."""
    from sklearn.model_selection import train_test_split as _tts
    assert abs(sum(fracs) - 1.0) < 1e-6, f"fracs must sum to 1.0, got {sum(fracs)}"
    remaining = np.asarray(ids)
    remaining_frac = 1.0
    groups = []
    for f in fracs[:-1]:
        this_size = f / remaining_frac
        group, remaining = _tts(remaining, train_size=this_size, random_state=seed)
        groups.append(group)
        remaining_frac -= f
    groups.append(remaining)
    return groups


# Guide v3.8 Sec 5.1's dev-pool table states ~70/~10/~5/~5 (of the 80% dev
# pool) for training/validation/calibration-selection/final-conformal-
# calibration -- these sum to 90%, not 100%, leaving 10 percentage points of
# the dev pool unassigned despite the guide's own text calling this a
# "partition" (which by definition must be exhaustive: Sec 2.4, "partitioned
# into four disjoint subsets"). Resolved by proportional rescaling, which
# preserves the stated 70:10:5:5 relative ratios while summing to exactly
# 100% -- documented in M2_Hybrid_Deviation_Log.md, not silently applied.
DEV_SPLIT_NAMES = ["training", "validation", "calibration_selection", "final_conformal_calibration"]
DEV_SPLIT_FRACTIONS = [70 / 90, 10 / 90, 5 / 90, 5 / 90]
DEV_SPLIT_PATH = os.path.join(DATA_DIR, "m2_hybrid_dev_pool_split.json")


def build_fixed_dev_split(force=False):
    """Builds (once) or loads (every subsequent call) the fixed, patient-
    level 4-way development-pool partition mandated by Guide v3.8 Sec
    5.1/Sec 7: a SINGLE random_state=42 split, reused identically by every
    seed and every model -- unlike the pre-v3.8 convention of re-splitting
    train/val per seed. Returns {name: {"subject_ids": np.ndarray,
    "sha256": str}} for the 4 DEV_SPLIT_NAMES. The 4 subsets are verified
    disjoint and exhaustive (covering data/train.csv's full subject_id set
    exactly) before being persisted -- this is not merely asserted.
    """
    if os.path.exists(DEV_SPLIT_PATH) and not force:
        with open(DEV_SPLIT_PATH) as f:
            saved = json.load(f)
        return {k: {"subject_ids": np.array(v["subject_ids"]), "sha256": v["sha256"]}
                for k, v in saved.items()}

    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), usecols=["subject_id"])
    dev_ids = np.sort(df_train["subject_id"].unique())
    groups = _split_ids(dev_ids, DEV_SPLIT_FRACTIONS, seed=42)

    result = {}
    for name, ids in zip(DEV_SPLIT_NAMES, groups):
        ids_sorted = np.sort(np.asarray(ids))
        result[name] = {"subject_ids": ids_sorted, "sha256": _hash_ids(ids_sorted)}

    all_ids = np.concatenate([v["subject_ids"] for v in result.values()])
    assert len(all_ids) == len(dev_ids), (
        f"split doesn't cover the full dev pool: {len(all_ids)} vs {len(dev_ids)}")
    assert len(np.unique(all_ids)) == len(dev_ids), "split has overlapping subjects across subsets"

    with open(DEV_SPLIT_PATH, "w") as f:
        json.dump({k: {"subject_ids": v["subject_ids"].tolist(), "sha256": v["sha256"]}
                   for k, v in result.items()}, f, indent=2)
    print(f"Built fixed dev-pool split (random_state=42): "
          + ", ".join(f"{k}={len(v['subject_ids'])}" for k, v in result.items()))
    return result


# ── Reproducibility manifest ("Implementation Fingerprint", Guide Sec 18) ───
MANIFEST_DIR = os.path.join(DATA_DIR, "m2_hybrid_manifests")


def save_manifest(model, seed, config):
    """Writes data/m2_hybrid_manifests/<model>_seed<N>.json: git commit
    hash, package versions, the seed, the full config dict, and the 4
    dev-pool-subset + test-set SHA-256 hashes. `config` should be a
    JSON-serializable dict of every hyperparameter/setting used for this
    run. Per project convention, this never creates a git commit itself --
    it only records whatever module 2's git history already is, honestly
    flagged when that history doesn't actually track this code (see
    M2_Hybrid_Deviation_Log.md)."""
    import subprocess
    import torch
    import mapie
    import xgboost
    import sklearn

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    module2_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    try:
        git_hash = subprocess.check_output(
            ["git", "-C", module2_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        git_note = ("module 2's .gitignore excludes this code (Benchmark_scripts/, data/); "
                    "this hash does not meaningfully version the run below.")
    except Exception:
        git_hash, git_note = None, "no git repository found at module 2 root"

    dev_split = build_fixed_dev_split()
    manifest = {
        "model": model, "seed": seed, "config": config,
        "git_commit_hash": git_hash, "git_note": git_note,
        "versions": {
            "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "mapie": mapie.__version__, "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dev_pool_subset_sha256": {k: v["sha256"] for k, v in dev_split.items()},
        "test_set_sha256": _hash_ids(load_fixed_test_subject_ids()),
    }
    safe_name = str(model).replace(" ", "_").replace("/", "_")
    with open(os.path.join(MANIFEST_DIR, f"{safe_name}_seed{seed}.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ── Per-patient prediction persistence (new -- no prior precedent in M2) ────
PREDICTIONS_PATH = os.path.join(DATA_DIR, "m2_hybrid_predictions.csv")


def save_predictions(subject_id, model, seed, y_true, y_prob,
                      pi_lo_mapie=None, pi_hi_mapie=None,
                      pi_lo_eci=None, pi_hi_eci=None, stay_id=None):
    """Appends one (model, seed)'s per-encounter predictions to the shared
    long-format file data/m2_hybrid_predictions.csv
    (subject_id, stay_id, model, seed, y_true, y_prob, pi_lo_mapie,
    pi_hi_mapie, pi_lo_eci, pi_hi_eci). Required for the Guide's
    patient-cluster bootstrap and the predictions_all_models.csv
    deliverable (Sec 8/18), which need paired per-encounter predictions
    across models -- every existing M2 script persists only aggregate
    metrics. `stay_id` (test.csv's unique per-encounter identifier) should
    always be passed -- `subject_id` alone repeats for the ~34% of
    patients with more than one ED visit, so it cannot alone distinguish
    one encounter's row from another within the same (model, seed) block.
    Rerunning a given (model, seed) replaces its prior rows rather than
    accumulating duplicates.
    """
    n = len(subject_id)

    def _col(x):
        return [np.nan] * n if x is None else x

    rows = pd.DataFrame({
        "subject_id": subject_id, "stay_id": _col(stay_id), "model": model, "seed": seed,
        "y_true": y_true, "y_prob": y_prob,
        "pi_lo_mapie": _col(pi_lo_mapie), "pi_hi_mapie": _col(pi_hi_mapie),
        "pi_lo_eci": _col(pi_lo_eci), "pi_hi_eci": _col(pi_hi_eci),
    })
    if os.path.exists(PREDICTIONS_PATH):
        existing = pd.read_csv(PREDICTIONS_PATH)
        existing = existing[~((existing["model"] == model) & (existing["seed"] == seed))]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows.to_csv(PREDICTIONS_PATH, index=False)


# ── Per-seed aggregate metrics (Guide Sec 18's named results_hybrid.csv) ────
RESULTS_HYBRID_PATH = os.path.join(DATA_DIR, "results_hybrid.csv")


def save_seed_metrics(model, seed, metrics: dict):
    """Appends one (model, seed)'s aggregate metrics as one row of
    data/results_hybrid.csv -- the Guide Sec 18 deliverable
    ("results_hybrid.csv with per-seed metrics"). `metrics` should carry
    whichever of auroc/auprc/brier/ece/coverage_mapie/width_mapie/
    coverage_eci/width_eci apply to this model (coverage/width only for
    Hybrid GRU-D and Frozen GRU-D, per Guide Sec 8). Same
    replace-on-rerun behavior as save_predictions().
    """
    row = {"model": model, "seed": seed, **metrics}
    if os.path.exists(RESULTS_HYBRID_PATH):
        existing = pd.read_csv(RESULTS_HYBRID_PATH)
        existing = existing[~((existing["model"] == model) & (existing["seed"] == seed))]
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(RESULTS_HYBRID_PATH, index=False)
