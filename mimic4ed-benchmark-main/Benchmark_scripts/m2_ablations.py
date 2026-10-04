"""
M2 Required Ablation Study
===========================
Roadmap requirements (SihhaDZ_Student_Roadmap_v8-1.pdf, M2 Experimental Protocol):
  1. ECI vs. MAPIE  (coverage @ 90%, interval width, latency)
  2. Prediction horizon  1h / 6h / 12h  (simulated via ED LOS filtering)
  3. Logit-transform vs. clipping  (bounded risk formulation)

Outputs:
  - Console tables + plots saved to Benchmark_scripts/m2_plots/
  - Appends ablation rows to data/m2_ablation_results.csv
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import json
from mapie.regression import SplitConformalRegressor
from m2_grud_cell import GRUDCell

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "m2_plots")
ECI_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Error-quantified-Conformal-Inference"))
os.makedirs(PLOT_DIR, exist_ok=True)

sys.path.insert(0, ECI_DIR)
from core.methods import ECI as eci_method

# The ECI paper's own reference configs (Error-quantified-Conformal-Inference/
# tests/configs/{elec2,GOOGL}_test.yaml) only ever validate ECI at
# lr in {0.05, 0.1, 0.5, 1.0}. This pipeline previously used lr=1.0 (top of the
# validated grid), chosen because the ~41-point demo test set needed the
# largest available step size to converge within so few updates. On the full
# 83,332-row credentialed test stream, m2_eci_lr_sweep.py swept the same grid
# and found lr=1.0 is in fact the *most conservative* point on it: average
# interval width fell 34.1% (0.4424 -> 0.2915) from lr=1.0 to lr=0.05 while
# empirical coverage only drifted 0.26 points (90.01% -> 89.75%), bringing ECI
# within 3.0% of MAPIE's width (vs. +56.3% at lr=1.0). lr=0.05 adopted as the
# primary full-scale configuration per supervisor sign-off; the full 4-point
# sweep (data/m2_eci_lr_sweep.csv) is retained as the ablation-section evidence
# for this choice, not silently discarded.
ECI_LR = 0.05

# ── Feature columns (identical to train_grud_conformal.py) ──────────────────
vitals_cols   = ['temperature','heartrate','resprate','o2sat','sbp','dbp','pain']
triage_vitals = [f'triage_{c}' for c in vitals_cols]
ed_vitals     = [f'ed_{c}_last' for c in vitals_cols]
static_cols   = [
    "age","gender","triage_acuity",
    "n_ed_30d","n_ed_90d","n_ed_365d",
    "n_hosp_30d","n_hosp_90d","n_hosp_365d",
    "n_icu_30d","n_icu_90d","n_icu_365d",
    "chiefcom_chest_pain","chiefcom_abdominal_pain","chiefcom_headache",
    "chiefcom_shortness_of_breath","chiefcom_back_pain","chiefcom_cough",
    "chiefcom_nausea_vomiting","chiefcom_fever_chills",
    "chiefcom_syncope","chiefcom_dizziness",
    "n_med","n_medrecon"
]
TARGET = "outcome_critical"
STATIC_SIZE = len(static_cols)  # 24

# ── Data helpers ─
# -- Track A scalers -------------------------------------------------------
# Fit on the SAME fixed dev-pool `training` subject subset the v3.8 pipeline
# uses (m2_hybrid_common.build_fixed_dev_split). That subset is a strict
# subset of train.csv, disjoint from both the MAPIE fit/calibration folds
# carved below and from test.csv, so this is leak-free for Track A too --
# and using one convention across both pipelines means there is a single
# scaler definition in this repo to report, not two competing ones.
_TRACKA_STATIC_SCALER = None


def _lazy_common():
    """Imported lazily, not at module level: m2_hybrid_common imports
    several helpers FROM this module, so a top-level import here would be a
    cycle. By the time any function below runs, both modules are fully
    initialised. (GRUDCell itself lives in m2_grud_cell.py precisely so the
    ARCHITECTURE needs no such workaround -- only these data helpers do.)"""
    import m2_hybrid_common as mhc
    return mhc


def _get_trackA_static_scaler():
    global _TRACKA_STATIC_SCALER
    if _TRACKA_STATIC_SCALER is not None:
        return _TRACKA_STATIC_SCALER
    mhc = _lazy_common()
    path = os.path.join(DATA_DIR, "m2_trackA_static_scaler.json")
    if os.path.exists(path):
        with open(path) as f:
            _TRACKA_STATIC_SCALER = json.load(f)
        return _TRACKA_STATIC_SCALER
    train_ids = set(mhc.build_fixed_dev_split()["training"]["subject_ids"].tolist())
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df = df[df["subject_id"].isin(train_ids)]
    S = _raw_static_frame(df)
    mean = np.nanmean(S, axis=0)
    sd = np.nanstd(S, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    _TRACKA_STATIC_SCALER = {"cols": list(static_cols),
                              "mean": [float(v) for v in mean],
                              "sd": [float(v) for v in sd],
                              "n_train_rows": int(len(df)),
                              "fit_on": "fixed dev-pool `training` subject subset only"}
    with open(path, "w") as f:
        json.dump(_TRACKA_STATIC_SCALER, f, indent=2)
    return _TRACKA_STATIC_SCALER


def _raw_static_frame(df):
    """The 24 Track A static features as a float array, before scaling."""
    df = df.copy()
    df['gender'] = df['gender'].map({'M':1,'F':0,'m':1,'f':0}).fillna(0).astype(int)
    for c in [c for c in static_cols if c not in df.columns]:
        df[c] = 0.0
    return df[static_cols].fillna(df[static_cols].median(numeric_only=True)).fillna(0).astype(np.float64).values


def prepare_data(df):
    """Packs (X | M | Delta | S) into one flat float32 array per encounter.

    Rewritten under M2_Hybrid_Deviation_Log.md Deviation 7 to bring Track A
    into line with the v3.8 pipeline. Three things changed, all of which
    previously applied to this file exactly as they did to the Hybrid one:

    1. Vitals are sourced from the CLEANED master lookup
       (m2_hybrid_common._load_master_vitals_lookup) rather than from
       train.csv/test.csv. Those files were median-imputed by
       data_general_processing.py before ever being written, so `M` was
       all-ones here too -- every Track A GRU-D result to date was produced
       with the mask carrying no information. The lookup preserves genuine
       missingness AND applies the pipeline's own Celsius conversion and
       outlier removal.
    2. Vitals and the 24 static features are standardised, fit on the
       training subset only. Previously both went in raw (temperature ~37,
       SBP ~131, visit counts, chief-complaint indicators) to a
       tanh/sigmoid GRU.
    3. Delta-t is clipped at 0 and expressed in units of the training-set
       median ED length of stay, matching the Hybrid pipeline exactly.
    """
    mhc = _lazy_common()
    sc = mhc.get_vitals_scaler()
    lookup = mhc._load_master_vitals_lookup()

    S_raw = _raw_static_frame(df)
    ssc = _get_trackA_static_scaler()
    assert ssc["cols"] == list(static_cols), "Track A static scaler column drift -- STOP."
    S = ((S_raw - np.asarray(ssc["mean"])) / np.asarray(ssc["sd"])).astype(np.float32)

    real = lookup.loc[df["stay_id"].values, triage_vitals + ed_vitals]
    mu = np.array([sc["mean"][c] for c in vitals_cols], dtype=np.float64)
    sd = np.array([sc["sd"][c] for c in vitals_cols], dtype=np.float64)
    T_raw = real[triage_vitals].values.astype(np.float64)
    E_raw = real[ed_vitals].values.astype(np.float64)
    T_m = (~np.isnan(T_raw)).astype(np.float64)
    E_m = (~np.isnan(E_raw)).astype(np.float64)
    T_v = np.nan_to_num((T_raw - mu) / sd, nan=0.0)
    E_v = np.nan_to_num((E_raw - mu) / sd, nan=0.0)

    if 'ed_los_hours' in df.columns:
        los = df['ed_los_hours'].astype(float)
        ed_los = los.fillna(los.median()).clip(lower=0.0).values
    else:
        ed_los = np.zeros(len(df))
    ed_los = ed_los / sc["delta_ref_hours"]

    X = np.stack([T_v, E_v], axis=1)
    M = np.stack([T_m, E_m], axis=1)
    d0 = np.zeros_like(T_m)
    d1 = np.tile(ed_los[:,None], (1,7))
    Delta = np.stack([d0, d1], axis=1).astype(np.float32)
    y = df[TARGET].astype(int).values
    N = X.shape[0]
    packed = np.concatenate([X.reshape(N,-1), M.reshape(N,-1), Delta.reshape(N,-1), S], axis=1).astype(np.float32)
    return packed, y


def patient_level_split(X, y, subjects, test_size, seed):
    """Splits X/y by unique subject_id, not by row. 34% of patients have more
    than one ED visit (established when the outer train/test split was fixed
    earlier this session), so a row-level train_test_split on X_train leaks
    the same patient's rows across both sides of a fit/calibration split --
    confirmed directly: 10,156 overlapping patients at seed=42 on the
    previous row-level version of this split. All rows for a given patient
    land on the same side of the split produced here."""
    from sklearn.model_selection import train_test_split as _tts
    unique_subj = np.unique(subjects)
    subj_a, subj_b = _tts(unique_subj, test_size=test_size, random_state=seed)
    mask_a = np.isin(subjects, subj_a)
    mask_b = np.isin(subjects, subj_b)
    return X[mask_a], X[mask_b], y[mask_a], y[mask_b], subjects[mask_a], subjects[mask_b]

def unpack(px):
    px = torch.tensor(px, dtype=torch.float32)
    N = px.shape[0]
    return px[:,:14].view(N,2,7), px[:,14:28].view(N,2,7), px[:,28:42].view(N,2,7), px[:,42:]

# ── GRU-D Model ─────────────────────────────────────────────────────────────
# NOTE on tunable params (clf_hidden, dropout): these change GRU-D's training
# procedure / classifier-head capacity only. The decay mechanism and the input
# feature set (inp=7 vitals x 2 timesteps, stat=24 static features) are the
# Gate-approved architecture and must never be changed by tuning -- see
# M2_Outcome_Definition_Gate.md and m2_grud_tuning.py's docstring.
#
# The recurrence itself now comes from the shared GRUDCell (m2_grud_cell.py),
# the same object the v3.8 pipeline uses. Under M2_Hybrid_Deviation_Log.md
# Deviation 7 this replaces the previous inline block, in which the decay
# term was multiplied by zero everywhere it appeared -- W_gamma/b_gamma
# received exactly zero gradient, stayed at their zero init for the whole of
# training, and Delta-t never reached the model at all. That defect was
# present in this Track A file exactly as it was in the Hybrid one, so every
# Track A GRU-D number predating this patch was produced by a masked GRU with
# zero-imputation rather than by GRU-D.
class GRUD(nn.Module):
    def __init__(self, inp=7, hid=16, stat=24, out=1, clf_hidden=32, dropout=0.0):
        super().__init__()
        self.hid = hid
        self.cell = GRUDCell(inp=inp, hid=hid)
        self.clf = nn.Sequential(
            nn.Linear(hid+stat, clf_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(clf_hidden, out)
        )
    def forward(self, X, M, D, S):
        h = self.cell(X, M, D)
        return self.clf(torch.cat([h,S],1))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SKLearnGRUD(BaseEstimator, ClassifierMixin):
    # Defaults below (pos_weight=15.75, hid=64) are the winning config from
    # m2_grud_tuning.py's 2026-09-01 validation-only re-search under
    # Deviation 7's corrected architecture (that script never opens
    # test.csv). Full trace: data/m2_grud_tuning_trace.csv.
    #
    # These REPLACE the previous defaults (hid=32, dropout=0.2,
    # weight_decay=1e-4), which are superseded twice over:
    #   1. They were selected against the pre-Deviation-7 pipeline, whose
    #      decay mechanism was inert and whose vitals were unscaled and in
    #      Fahrenheit -- a search over a model that was not the model.
    #   2. The first re-run of that search was itself invalid, because
    #      rolling those defaults into this constructor turned every
    #      unspecified-kwargs "baseline" arm INTO the previous winner. Stage 3
    #      compared dropout=0.2 against a baseline that already had
    #      dropout=0.2 and returned bit-identical numbers for all three arms.
    #      m2_grud_tuning.py now pins an explicit UNTUNED_BASE into every
    #      candidate so this constructor's defaults can never contaminate a
    #      future search again. See Deviation 8.
    #
    # pos_weight=15.75 won that search on validation AUROC and is NOT adopted.
    # AUROC is invariant to any monotone rescaling of the scores, so the search
    # is structurally blind to decalibration -- and m2_posweight_calibration_check.py
    # (4 arms x 5 seeds, each config passed explicitly) shows pos_weight is the
    # sole cause of a severe one:
    #
    #   hid=16, no pos_weight : AUROC 0.9151  ECE 0.0017  mean_p 0.0597  ECI width 0.2619
    #   hid=64, no pos_weight : AUROC 0.9221  ECE 0.0020  mean_p 0.0597  ECI width 0.2432  <- adopted
    #   hid=16, pos_weight    : AUROC 0.9251  ECE 0.1863  mean_p 0.2451  ECI width 0.7754
    #   hid=64, pos_weight    : AUROC 0.9259  ECE 0.1781  mean_p 0.2368  ECI width 0.7584
    #
    # pos_weight buys +0.004 AUROC over the adopted arm while inflating ECE ~110x
    # and widening the conformal interval ~3.1x, and it drives the mean predicted
    # probability to 0.2368 against a true test base rate of 0.0588 -- a risk score
    # asserting ~4x the real event rate. Coverage stays at nominal throughout, which
    # is exactly why coverage alone cannot be used to check this: split-conformal and
    # ECI both hold marginal coverage by paying in WIDTH. M2's Track A contribution
    # is the interval, so that trade is rejected.
    #
    # hid=64 is adopted: it improves AUROC, AUPRC, Brier AND width simultaneously
    # over hid=16, at equal calibration. See Deviation 8.
    def __init__(self, epochs=150, lr=0.01, seed=42, hid=64, clf_hidden=32,
                 dropout=0.0, pos_weight=None, weight_decay=0.0, batch_size=None,
                 early_stopping_patience=None):
        # batch_size=None is the tuned config's full-batch behaviour; the
        # re-search found batch_size=2048 now clearly harmful (0.8830).
        # early_stopping_patience=None reproduces the tuned config's
        # fixed-epochs behavior (requires X_val/y_val passed to fit()).
        self.epochs, self.lr, self.seed = epochs, lr, seed
        self.hid, self.clf_hidden, self.dropout = hid, clf_hidden, dropout
        self.pos_weight, self.weight_decay, self.batch_size = pos_weight, weight_decay, batch_size
        self.early_stopping_patience = early_stopping_patience
        self.model, self.classes_ = None, np.array([0,1])
    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.model = GRUD(stat=STATIC_SIZE, hid=self.hid, clf_hidden=self.clf_hidden,
                           dropout=self.dropout).to(DEVICE)
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pw = torch.tensor([self.pos_weight], device=DEVICE) if self.pos_weight is not None else None
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        Xt,Mt,Dt,St = unpack(X)
        Xt,Mt,Dt,St = Xt.to(DEVICE),Mt.to(DEVICE),Dt.to(DEVICE),St.to(DEVICE)
        yt = torch.tensor(y, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        n = Xt.shape[0]

        use_early_stop = self.early_stopping_patience is not None and X_val is not None
        best_val_auroc, patience_left, best_state = -1.0, self.early_stopping_patience, None

        def maybe_early_stop():
            nonlocal best_val_auroc, patience_left, best_state
            if not use_early_stop:
                return False
            self.model.eval()
            with torch.no_grad():
                Xv,Mv,Dv,Sv = unpack(X_val)
                Xv,Mv,Dv,Sv = Xv.to(DEVICE),Mv.to(DEVICE),Dv.to(DEVICE),Sv.to(DEVICE)
                p_val = torch.sigmoid(self.model(Xv,Mv,Dv,Sv)).cpu().numpy().ravel()
            from sklearn.metrics import roc_auc_score as _auc
            val_auroc = _auc(y_val, p_val)
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_left = self.early_stopping_patience
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_left -= 1
            return patience_left <= 0

        if self.batch_size is None:
            # Original full-batch loop, byte-for-byte unchanged when early
            # stopping is not requested.
            for epoch in range(self.epochs):
                self.model.train(); opt.zero_grad()
                loss = crit(self.model(Xt,Mt,Dt,St), yt); loss.backward(); opt.step()
                if use_early_stop and maybe_early_stop():
                    break
        else:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            for epoch in range(self.epochs):
                self.model.train()
                perm = torch.randperm(n, generator=gen)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i+self.batch_size]
                    opt.zero_grad()
                    loss = crit(self.model(Xt[idx],Mt[idx],Dt[idx],St[idx]), yt[idx])
                    loss.backward(); opt.step()
                if use_early_stop and maybe_early_stop():
                    break
        if use_early_stop and best_state is not None:
            self.model.load_state_dict(best_state)
        return self
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt,Mt,Dt,St = unpack(X)
            Xt,Mt,Dt,St = Xt.to(DEVICE),Mt.to(DEVICE),Dt.to(DEVICE),St.to(DEVICE)
            p = torch.sigmoid(self.model(Xt,Mt,Dt,St)).cpu().numpy()
        return np.hstack([1-p, p])
    def predict(self, X):
        return (self.predict_proba(X)[:,1] > 0.5).astype(int)

class GRUDRiskRegressor(BaseEstimator, RegressorMixin):
    """Exposes the GRU-D positive-class probability as a continuous regression
    target so MAPIE's split-conformal regressor calibrates a genuine numeric
    interval on the risk score in [0, 1] (roadmap M2 core: "Risk Score [0, 1]
    + MAPIE prediction interval"), directly comparable to ECI's interval
    rather than a discrete classification prediction set."""
    def __init__(self, clf):
        self.clf = clf
    def fit(self, X, y):
        return self
    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]

# ── Conformal helpers ────────────────────────────────────────────────────────
def _sorted_bounds(lo, hi):
    """Guarantee lo <= hi before clipping to [0,1].

    ECI's tracked q is a signed online quantity by design and routinely goes
    negative; the reference implementation's own driver (tests/base_test.py)
    sorts the two constructed bounds before use (its `minsize` floor, default
    0) so a negative q degrades to a zero-width interval instead of an
    inverted one. An inverted interval can never contain the true value, so
    without this fix every such step is silently scored as "not covered"
    regardless of the actual outcome — this was the primary cause of ECI's
    coverage collapse.
    """
    return np.minimum(lo, hi), np.maximum(lo, hi)

def run_mapie(model, X_cal, y_cal, X_test, y_test):
    """Back-compatible 4-tuple wrapper around run_mapie_bounds(); existing
    callers unpack exactly (y_prob, coverage, width, latency)."""
    y_prob, cov, wid, lat, _lo, _hi = run_mapie_bounds(model, X_cal, y_cal, X_test, y_test)
    return y_prob, cov, wid, lat


def run_mapie_bounds(model, X_cal, y_cal, X_test, y_test):
    """As run_mapie(), but also returns the per-encounter interval bounds,
    which class-conditional coverage reporting needs (a marginal coverage
    number cannot be decomposed after the fact)."""
    reg = GRUDRiskRegressor(model)
    mapie_reg = SplitConformalRegressor(estimator=reg, confidence_level=0.9, prefit=True)
    mapie_reg.conformalize(X_cal, y_cal.astype(float))
    t0 = time.perf_counter()
    y_pred, y_pis = mapie_reg.predict_interval(X_test)
    lat = (time.perf_counter()-t0)*1000/len(X_test)
    lo, hi = _sorted_bounds(y_pis[:,0,0], y_pis[:,1,0])
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    y_prob = model.predict_proba(X_test)[:,1]
    cov = np.mean((y_test >= lo) & (y_test <= hi))
    wid = np.mean(hi - lo)
    return y_prob, cov, wid, lat, lo, hi

def run_eci(model, X_test_sorted, y_test_sorted, scores=None):
    y_prob = model.predict_proba(X_test_sorted)[:,1]
    if scores is None:
        scores = np.abs(y_test_sorted - y_prob)
    t0 = time.perf_counter()
    res = eci_method(scores=scores, alpha=0.1, lr=ECI_LR, T_burnin=5, ahead=1, proportional_lr=True)
    lat = (time.perf_counter()-t0)*1000/len(X_test_sorted)
    qs = res['q']
    lo, hi = _sorted_bounds(y_prob - qs, y_prob + qs)
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    cov = np.mean((y_test_sorted >= lo) & (y_test_sorted <= hi))
    wid = np.mean(hi - lo)
    return y_prob, cov, wid, lat, lo, hi

def run_eci_logit(model, X_test_sorted, y_test_sorted):
    """ECI with the final interval built in logit space then mapped back to
    [0,1] — roadmap requirement. Calibrated on the same score as run_eci
    (|y - phat|, bounded in [0,1]); only the bounding differs. The label
    itself is never logit-transformed: logit(0)/logit(1) are +/-infinity, so
    doing that would replace the error signal with a near-constant ~13.8
    offset that just re-encodes the class label instead of measuring
    prediction error, and (via ECI's proportional_lr) blow up the tracker's
    step size and destabilize q."""
    y_prob = model.predict_proba(X_test_sorted)[:,1]
    eps = 1e-6
    scores = np.abs(y_test_sorted - y_prob)
    res = eci_method(scores=scores, alpha=0.1, lr=ECI_LR, T_burnin=5, ahead=1, proportional_lr=True)
    qs = res['q']
    y_prob_c = np.clip(y_prob, eps, 1-eps)
    logit_prob = np.log(y_prob_c/(1-y_prob_c))
    lo_logit, hi_logit = _sorted_bounds(logit_prob - qs, logit_prob + qs)
    lo = 1/(1+np.exp(-lo_logit))
    hi = 1/(1+np.exp(-hi_logit))
    cov = np.mean((y_test_sorted >= lo) & (y_test_sorted <= hi))
    wid = np.mean(hi - lo)
    return cov, wid, lo, hi

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # ECI is evaluated as an online stream, so test order matters. MIMIC-IV
    # shifts dates by a random per-patient offset (anchor_year spans
    # 2110-2208 in this cohort), so sorting by raw `intime` orders patients
    # arbitrarily while looking chronological. sort_chronologically() uses
    # the de-shifted approximate real year instead -- see
    # m2_hybrid_common._load_real_time_lookup() for the recipe and its
    # +/-1 year resolution limit.
    df_test = _lazy_common().sort_chronologically(df_test)

    X_train, y_train = prepare_data(df_train)
    X_test,  y_test  = prepare_data(df_test)
    train_subjects = df_train["subject_id"].values

    # Split-conformal's coverage guarantee assumes the calibration set is
    # exchangeable with, and disjoint from, the set the model was fit on.
    # Hold out a stratified calibration fold from the training pool for MAPIE
    # rather than reusing the fit data (whose residuals are optimistically
    # small). ECI is unaffected: it calibrates online on the test stream it's
    # evaluated on, which is correct by construction for an adaptive method.
    # Patient-level split (not row-level): 34% of patients have >1 ED visit,
    # so a row-level split leaks the same patient across fit and cal --
    # confirmed directly (10,156 overlapping patients at seed=42 before this
    # fix). See patient_level_split() above.
    X_fit, X_cal, y_fit, y_cal, _, _ = patient_level_split(
        X_train, y_train, train_subjects, test_size=0.2, seed=42
    )

    print("Training GRU-D (shared model for all ablations)...")
    model = SKLearnGRUD(epochs=150, lr=0.01)
    model.fit(X_fit, y_fit)

    auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    auprc = average_precision_score(y_test, model.predict_proba(X_test)[:,1])

    rows = []

    # ─────────────────────────────────────────────────────────────────────────
    # ABLATION 1: MAPIE vs ECI (coverage, width, latency)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ABLATION 1: MAPIE vs ECI")
    print("="*60)

    y_prob_m, cov_m, wid_m, lat_m = run_mapie(model, X_cal, y_cal, X_test, y_test)
    y_prob_e, cov_e, wid_e, lat_e, lo_e, hi_e = run_eci(model, X_test, y_test)

    print(f"\n{'Method':<20} {'AUROC':>8} {'AUPRC':>8} {'Coverage':>10} {'Width':>8} {'Lat(ms)':>10}")
    print("-"*66)
    print(f"{'GRU-D + MAPIE':<20} {auroc:>8.4f} {auprc:>8.4f} {cov_m:>10.4f} {wid_m:>8.4f} {lat_m:>10.4f}")
    print(f"{'GRU-D + ECI':<20} {auroc:>8.4f} {auprc:>8.4f} {cov_e:>10.4f} {wid_e:>8.4f} {lat_e:>10.4f}")

    rows.append({'ablation':'MAPIE_vs_ECI','variant':'GRU-D + MAPIE','auroc':round(auroc,4),'auprc':round(auprc,4),'coverage':round(cov_m,4),'width':round(wid_m,4),'latency_ms':round(lat_m,4)})
    rows.append({'ablation':'MAPIE_vs_ECI','variant':'GRU-D + ECI' ,'auroc':round(auroc,4),'auprc':round(auprc,4),'coverage':round(cov_e,4),'width':round(wid_e,4),'latency_ms':round(lat_e,4)})

    # Plot 1: ROC curves (both use same model, so one ROC; showing interval widths)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Panel A: Interval width comparison bar chart
    axes[0].bar(['MAPIE','ECI'], [wid_m, wid_e], color=['#4C72B0','#DD8452'], edgecolor='black')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Max possible width')
    axes[0].set_ylabel('Average Interval Width'); axes[0].set_title('A) Interval Width: MAPIE vs ECI')
    axes[0].set_ylim(0, 1.2); axes[0].legend()
    # Panel B: Coverage comparison
    axes[1].bar(['MAPIE','ECI'], [cov_m*100, cov_e*100], color=['#4C72B0','#DD8452'], edgecolor='black')
    axes[1].axhline(y=90, color='red', linestyle='--', label='Target 90%')
    axes[1].set_ylabel('Empirical Coverage (%)'); axes[1].set_title('B) Coverage: MAPIE vs ECI')
    axes[1].set_ylim(0, 110); axes[1].legend()
    plt.suptitle('Ablation 1: MAPIE vs ECI (MIMIC-IV-ED demo subset)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ablation1_mapie_vs_eci.png'), dpi=150)
    plt.close()
    print(f"  → Saved: ablation1_mapie_vs_eci.png")

    # ─────────────────────────────────────────────────────────────────────────
    # ABLATION 2: Prediction horizon 1h / 6h / 12h
    # Strategy: filter test patients by ED LOS to simulate early prediction windows
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ABLATION 2: Prediction Horizon (1h / 6h / 12h)")
    print("="*60)
    print("  (Simulated via ED LOS filtering: patients discharged within Xh")
    print("   represent the cohort visible at that prediction horizon)")

    horizon_results = []
    if 'ed_los_hours' in df_test.columns:
        for h_label, h_max in [('1h', 1.0), ('6h', 6.0), ('12h', 12.0), ('All', 9999.0)]:
            mask = df_test['ed_los_hours'].fillna(9999) <= h_max
            df_h = df_test[mask]
            if len(df_h) < 5 or df_h[TARGET].nunique() < 2:
                print(f"  Horizon {h_label}: not enough data (n={len(df_h)}, skipping).")
                horizon_results.append({'horizon':h_label,'n':len(df_h),'auroc':'N/A','auprc':'N/A','coverage_mapie':'N/A','coverage_eci':'N/A'})
                continue
            Xh, yh = prepare_data(df_h)
            yph = model.predict_proba(Xh)[:,1]
            auroc_h = roc_auc_score(yh, yph)
            auprc_h = average_precision_score(yh, yph)
            # MAPIE calibrated on full train (continuous interval, same helper as Ablation 1)
            _, cov_mh, _, _ = run_mapie(model, X_cal, y_cal, Xh, yh)
            # ECI (same helper as Ablation 1; T_burnin shortened for small horizon cohorts)
            if len(yh) >= 10:
                _, cov_eh, _, _, _, _ = run_eci(model, Xh, yh, scores=np.abs(yh - yph))
            else:
                cov_eh = float('nan')
            print(f"  Horizon {h_label:>4}: n={len(df_h):>4}  AUROC={auroc_h:.4f}  AUPRC={auprc_h:.4f}  Cov_MAPIE={cov_mh:.4f}  Cov_ECI={cov_eh:.4f}")
            horizon_results.append({'horizon':h_label,'n':len(df_h),'auroc':round(auroc_h,4),'auprc':round(auprc_h,4),'coverage_mapie':round(cov_mh,4),'coverage_eci':round(cov_eh,4) if not np.isnan(cov_eh) else 'N/A'})
            rows.append({'ablation':'Horizon','variant':f'Horizon {h_label}','auroc':round(auroc_h,4),'auprc':round(auprc_h,4),'coverage':round(cov_mh,4),'coverage_eci':round(cov_eh,4) if not np.isnan(cov_eh) else float('nan'),'width':float('nan'),'latency_ms':float('nan')})
    else:
        print("  'ed_los_hours' column not found — cannot filter by horizon.")
        horizon_results = [{'horizon':'N/A','n':len(df_test),'auroc':round(auroc,4),'auprc':round(auprc,4),'coverage_mapie':round(cov_m,4),'coverage_eci':round(cov_e,4)}]

    # Plot 2: AUROC by horizon
    valid_h = [r for r in horizon_results if r['auroc'] != 'N/A']
    if valid_h:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [r['horizon'] for r in valid_h]
        aurocs = [r['auroc'] for r in valid_h]
        auprc_vals = [r['auprc'] for r in valid_h]
        x = np.arange(len(labels))
        ax.bar(x-0.2, aurocs, 0.35, label='AUROC', color='#4C72B0', edgecolor='black')
        ax.bar(x+0.2, auprc_vals, 0.35, label='AUPRC', color='#55A868', edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel('Prediction Horizon'); ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05); ax.legend()
        ax.set_title('Ablation 2: GRU-D Performance by Prediction Horizon\n(MIMIC-IV-ED demo subset)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'ablation2_prediction_horizons.png'), dpi=150)
        plt.close()
        print(f"  → Saved: ablation2_prediction_horizons.png")

    # ─────────────────────────────────────────────────────────────────────────
    # ABLATION 3: Logit-transform vs Clipping
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ABLATION 3: Logit-Transform vs Clipping (ECI interval bounding)")
    print("="*60)

    # Clipping (already computed as cov_e, wid_e)
    cov_clip, wid_clip = cov_e, wid_e

    # Logit-transform
    cov_logit, wid_logit, lo_logit, hi_logit = run_eci_logit(model, X_test, y_test.astype(float))

    print(f"\n{'Bounding Method':<25} {'Coverage':>10} {'Width':>8}")
    print("-"*45)
    print(f"{'ECI + Clipping':<25} {cov_clip:>10.4f} {wid_clip:>8.4f}")
    print(f"{'ECI + Logit-transform':<25} {cov_logit:>10.4f} {wid_logit:>8.4f}")

    rows.append({'ablation':'Logit_vs_Clip','variant':'ECI + Clipping',       'auroc':round(auroc,4),'auprc':round(auprc,4),'coverage':round(cov_clip,4),'width':round(wid_clip,4),'latency_ms':float('nan')})
    rows.append({'ablation':'Logit_vs_Clip','variant':'ECI + Logit-transform','auroc':round(auroc,4),'auprc':round(auprc,4),'coverage':round(cov_logit,4),'width':round(wid_logit,4),'latency_ms':float('nan')})

    # Plot 3: Side-by-side coverage and width for clipping vs logit
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    methods = ['Clipping', 'Logit-Transform']
    covs = [cov_clip*100, cov_logit*100]
    wids = [wid_clip, wid_logit]
    axes[0].bar(methods, covs, color=['#4C72B0','#C44E52'], edgecolor='black')
    axes[0].axhline(y=90, color='red', linestyle='--', label='Target 90%')
    axes[0].set_ylabel('Empirical Coverage (%)'); axes[0].set_title('A) Coverage')
    axes[0].set_ylim(0, 110); axes[0].legend()
    axes[1].bar(methods, wids, color=['#4C72B0','#C44E52'], edgecolor='black')
    axes[1].set_ylabel('Average Interval Width'); axes[1].set_title('B) Interval Width')
    axes[1].set_ylim(0, 1.2)
    plt.suptitle('Ablation 3: ECI Bounding — Clipping vs Logit-Transform\n(MIMIC-IV-ED demo subset)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ablation3_logit_vs_clipping.png'), dpi=150)
    plt.close()
    print(f"  → Saved: ablation3_logit_vs_clipping.png")

    # Bonus: ROC curve plot
    from sklearn.metrics import roc_curve
    y_prob_all = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob_all)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, color='#4C72B0', lw=2, label=f'GRU-D (AUROC = {auroc:.4f})')
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('GRU-D ROC Curve — Task 2 Critical Outcome\n(MIMIC-IV-ED demo subset)')
    ax.legend(); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'roc_curve_grud.png'), dpi=150)
    plt.close()
    print(f"\n  → Saved: roc_curve_grud.png")

    # ─────────────────────────────────────────────────────────────────────────
    # Save ablation table to CSV
    # ─────────────────────────────────────────────────────────────────────────
    abl_df = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "m2_ablation_results.csv")
    abl_df.to_csv(out_path, index=False)
    print(f"\nAll ablation results saved to: data/m2_ablation_results.csv")
    print(f"All plots saved to:            Benchmark_scripts/m2_plots/")
    print("\nM2 ablation study complete.")
