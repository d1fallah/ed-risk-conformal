"""
Phase 2 shared loaders and feature builders. Row-level files live in data/revision_v2/ (untracked); nothing row-level is printed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402

OUTD = os.path.join(C.DATA, "revision_v2")
PRED_DIR = os.environ.get("P2_PRED_DIR", os.path.join(OUTD, "preds"))
CKPT_DIR = os.path.join(OUTD, "checkpoints")
RUN_DIR = os.path.join(OUTD, "runs")
for _d in (PRED_DIR, CKPT_DIR, RUN_DIR):
    os.makedirs(_d, exist_ok=True)
ROLES = ["fit", "val", "cs", "fc", "test"]
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
STATIC = ["age", "sex", "triage_acuity", "comorbidity_elixhauser_count"]
T_ED_H2 = 7


def load_cohort(lm, outcome="primary"):
    """Eligible stays of a landmark with labels, split roles and (fixed) static features.
    outcome 'primary' -> y = y_primary on lm_eligible; 'secondary' -> y_secondary on lm_eligible_secondary."""
    df = pd.read_csv(os.path.join(OUTD, f"{lm}_cohort.csv"), parse_dates=["intime", "outtime"])
    fix_path = os.path.join(OUTD, "elix_fix.csv")
    if os.path.exists(fix_path):
        fx = pd.read_csv(fix_path).set_index("stay_id")["count_fixed"]
        m = df["stay_id"].isin(fx.index)
        df.loc[m, "comorbidity_elixhauser_count"] = fx.loc[df.loc[m, "stay_id"]].values
    if outcome == "primary":
        df = df[df["lm_eligible"]].copy()
        df["y"] = df["y_primary"].astype(int)
    else:
        df = df[df["lm_eligible_secondary"]].copy()
        df["y"] = df["y_secondary"].astype(int)
    df["split_name"] = np.where(df["split_role"] == "test", "test_v2", "train_v2")
    return df.reset_index(drop=True)


def role(df, r):
    return df[df["split_role"] == r].reset_index(drop=True)


def h2_arrays(df):
    """(values, obs_mask, pad_mask, delta_hours, t_hours) for the rows of df, looked up by stay_id from the 2 h tensors."""
    parts = {}
    for sp in ("train_v2", "test_v2"):
        sub = df[df["split_name"] == sp]
        if len(sub) == 0:
            continue
        with np.load(os.path.join(OUTD, f"h2_sequences_{sp}.npz")) as z:
            pos = pd.Series(np.arange(len(z["stay_id"])), index=z["stay_id"]).loc[sub["stay_id"].values].values
            parts[sp] = {k: z[k][pos] for k in ("values", "obs_mask", "pad_mask", "delta_hours", "t_hours")}
        parts[sp]["idx"] = sub.index.values
    out = {k: np.zeros((len(df),) + parts[next(iter(parts))][k].shape[1:], dtype=np.float32) for k in
           ("values", "obs_mask", "pad_mask", "delta_hours", "t_hours")}
    for sp, d in parts.items():
        for k in out:
            out[k][d["idx"]] = d[k]
    return out["values"], out["obs_mask"], out["pad_mask"], out["delta_hours"], out["t_hours"]


def triage_arrays(df):
    """Triage vitals (n, 7) and observed mask from slot 0 of the original tensors."""
    vals = np.full((len(df), 7), np.nan, dtype=np.float32)
    obs = np.zeros((len(df), 7), dtype=np.float32)
    for sp in ("train_v2", "test_v2"):
        sub = df[df["split_name"] == sp]
        if len(sub) == 0:
            continue
        v, o, _p, _d, _t = C.sequences(sp, sub["stay_id"].values)
        vals[sub.index.values] = v[:, 0, :]
        obs[sub.index.values] = o[:, 0, :]
    return vals, obs


def static_matrix(df):
    return df[STATIC].astype(float).values


# ----------------------------------------------------------------------------------------------- features
def aggregate_features(values, obs, pad):
    """Matched aggregate features over the triage slot and the ED slots (mean, min, max, last, count per vital)."""
    n, T, V = values.shape
    m = obs.astype(bool)
    f = {}
    for i, v in enumerate(VITALS):
        col = np.where(m[:, :, i], values[:, :, i], np.nan)
        with np.errstate(all="ignore"):
            f[f"{v}_mean"] = np.nanmean(col, axis=1)
            f[f"{v}_min"] = np.nanmin(col, axis=1)
            f[f"{v}_max"] = np.nanmax(col, axis=1)
        f[f"{v}_count"] = m[:, :, i].sum(axis=1).astype(float)
        last = np.full(n, np.nan)
        anyo = m[:, :, i].any(axis=1)
        idx = np.where(m[:, :, i], np.arange(T), -1).max(axis=1)
        last[anyo] = col[np.where(anyo)[0], idx[anyo]]
        f[f"{v}_last"] = last
    return pd.DataFrame(f)


def timing_features(values, obs, pad, t_hours, L_hours=2.0):
    """Timing-aware additions (plan 2.3): time span, inter-measurement interval statistics, time since the last measurement,
    missingness counts. Times are hours after arrival; the triage slot is time 0."""
    n, T, V = values.shape
    ed = pad.copy(); ed[:, 0] = 0
    ed_b = ed.astype(bool)
    t = np.where(ed_b, t_hours, np.nan)
    f = {}
    n_ed = ed.sum(axis=1)
    f["n_ed_rows"] = n_ed
    with np.errstate(all="ignore"):
        first = np.nanmin(t, axis=1); last = np.nanmax(t, axis=1)
        f["first_ed_row_time"] = first
        f["last_ed_row_time"] = last
        f["time_span_ed_rows"] = last - first
        gaps = np.diff(t, axis=1)                       # NaN where either neighbour is padding
        f["gap_mean"] = np.nanmean(gaps, axis=1)
        f["gap_max"] = np.nanmax(gaps, axis=1)
        f["gap_min"] = np.nanmin(gaps, axis=1)
        f["gap_std"] = np.nanstd(gaps, axis=1)
        any_t = np.where(np.nan_to_num(last, nan=0.0) > 0, last, 0.0)           # triage at time 0 counts as a measurement
        f["time_since_last_measurement"] = L_hours - any_t
    for i, v in enumerate(VITALS):
        o = obs[:, :, i].astype(bool)
        tt = np.where(o, t_hours, np.nan)
        tt[:, 0] = np.where(o[:, 0], 0.0, np.nan)
        with np.errstate(all="ignore"):
            lastv = np.nanmax(tt, axis=1)
        f[f"{v}_time_since_last"] = L_hours - lastv
        f[f"{v}_missing_ed_rows"] = (ed_b & ~o).sum(axis=1).astype(float)
    cells = (ed_b[:, :, None] & np.ones((1, 1, V), bool)).sum(axis=(1, 2)).astype(float)
    miss = (ed_b[:, :, None] & ~obs.astype(bool)).sum(axis=(1, 2)).astype(float)
    f["missing_cells_ed"] = miss
    f["missing_fraction_ed"] = np.where(cells > 0, miss / np.maximum(cells, 1), np.nan)
    f["triage_missing_count"] = (~obs[:, 0, :].astype(bool)).sum(axis=1).astype(float)
    return pd.DataFrame(f)


def h2_feature_frame(df, timing):
    values, obs, pad, _delta, th = h2_arrays(df)
    X = aggregate_features(values, obs, pad)
    if timing:
        X = pd.concat([X, timing_features(values, obs, pad, th)], axis=1)
    S = pd.DataFrame(static_matrix(df), columns=STATIC)
    return pd.concat([X, S], axis=1)


def triage_feature_frame(df):
    vals, obs = triage_arrays(df)
    X = pd.DataFrame({f"triage_{v}": np.where(obs[:, i] > 0, vals[:, i], np.nan) for i, v in enumerate(VITALS)})
    S = pd.DataFrame(static_matrix(df), columns=STATIC)
    return pd.concat([X, S], axis=1)


def save_preds(name, seed, df, score):
    np.savez_compressed(os.path.join(PRED_DIR, f"{name}_seed{seed}.npz"), stay_id=df["stay_id"].values, score=np.asarray(score, dtype=np.float64))


def load_preds(name, seed):
    path = os.path.join(PRED_DIR, f"{name}_seed{seed}.npz")
    if not os.path.exists(path):          # deterministic models (logistic regression) are stored once, under seed 42
        path = os.path.join(PRED_DIR, f"{name}_seed42.npz")
    z = np.load(path)
    return z["stay_id"], z["score"]
