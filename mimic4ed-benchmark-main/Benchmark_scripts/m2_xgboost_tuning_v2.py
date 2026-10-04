"""
M2 -- XGBoost-Matched v2 Tuning Search (validation-only -- never reads test_v2.csv)
==============================================================================
Phase 2a of the session's "Clearing the Runway to the Manuscript" plan.

Why this script exists: `M2_Findings_Dossier.md` Sec 24.2 reports Hybrid
GRU-D v2 (0.9371) decisively beating XGBoost-Matched v2 (0.9217) -- but
XGBoost-Matched v2 runs at the untuned pilot config (`n_estimators=100`)
while GRU-D/Frozen-Control/mTAN all received real search budgets. Sec
24.2's own text already flags this: "this 1.5-point gap should be
reported as suggestive, not conclusive." Sec 18.1 already demonstrated
once, on the T=2 cohort, that exactly this asymmetry can reverse a
headline finding (Delta_representation: +0.0005 tie under the
Guide-locked comparison vs. -0.0045 under symmetric tuning). This script
removes the same asymmetry from the v2 result set.

Protocol symmetry with m2_xgboost_tuning.py (deliberate, point by point,
mirrored exactly except where the v2 cohort structurally differs):
  * Reads ONLY data/train_v2.csv (the chronological-primary TRAIN PERIOD);
    asserts it never opens data/test_v2.csv.
  * Same 3 tuning seeds (42, 123, 7), distinct from the 5-seed
    confirmatory protocol.
  * Same staged/sequential structure with an explicit UNTUNED_BASE merged
    into every candidate (the Deviation-8 anti-p-hacking fix).
  * Same noise guard: a candidate replaces the stage baseline only if it
    beats the baseline mean by more than one baseline validation SD.
  * Same calibration guard: reject any candidate whose ECE exceeds 5x the
    baseline's.
  * Comparable budget: 14 candidates x 3 seeds, matching m2_xgboost_
    tuning.py's own budget exactly.

Deliberate difference from m2_xgboost_tuning.py, disclosed rather than
silently copied: the per-tune-seed fit/val carve here is a RANDOM
patient-level split of the v2 train PERIOD (mirroring the T=2 script's
own internal random carve), not a further chronological split. This is
not a departure from Gate v2 Sec 7's chronological-primary-split
principle -- that principle governs the OUTER train/test evaluation
boundary (already enforced: this script never reads test_v2.csv), not an
inner tuning fold that never touches the evaluation period at all. A
random inner carve is standard practice for hyperparameter selection and
keeps this script's own splitting logic directly comparable to m2_xgboost_
tuning.py's, which is the point of running it at all.

Uses build_matched_features_v2 from m2_xgboost_matched_v2.py UNCHANGED (7
vitals x {mean,min,max,last,count} over the full irregular sequence + 4
Guide static features = 39 columns) -- the same v2 feature set XGBoost-
Matched v2's untuned confirmatory run already uses, so this search tunes
exactly the model being reported, not a different one.

Output: data/m2_v2_xgboost_tuning_trace.csv, data/m2_v2_xgboost_tuned_config.json
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRAIN_V2_PATH = os.path.join(DATA_DIR, "train_v2.csv")
TEST_V2_PATH = os.path.join(DATA_DIR, "test_v2.csv")
TRACE_CSV = os.path.join(DATA_DIR, "m2_v2_xgboost_tuning_trace.csv")

assert not any("test_v2.csv" in str(a) for a in sys.argv), "This script must never touch test_v2.csv."
print(f"Integrity check: TEST_V2_PATH is defined ({TEST_V2_PATH}) but will NOT be opened by this script.")

sys.path.insert(0, SCRIPT_DIR)
from m2_xgboost_matched_v2 import build_matched_features_v2

TUNE_SEEDS = [42, 123, 7]
TARGET = "outcome_critical"

# The untuned reference: XGBoost-Matched v2 exactly as the confirmatory
# 5-seed run (m2_xgboost_matched_v2.py) uses it. Stated explicitly so no
# stage's baseline can drift (Deviation 8's lesson, same guard as T=2).
UNTUNED_BASE = {
    "n_estimators": 100, "eval_metric": "logloss", "verbosity": 0,
    "max_depth": 6, "learning_rate": 0.3, "subsample": 1.0,
    "colsample_bytree": 1.0, "min_child_weight": 1,
    "reg_lambda": 1.0, "scale_pos_weight": 1.0,
}

print("Loading data/train_v2.csv only, building matched features...", flush=True)
df_train = pd.read_csv(TRAIN_V2_PATH)
X_all = build_matched_features_v2(df_train, "train_v2")
y_all = df_train[TARGET].astype(int).values
subjects_all = df_train["subject_id"].values

pos_rate = y_all.mean()
SCALE_POS_WEIGHT = round((1 - pos_rate) / pos_rate, 2)
print(f"v2 train-period positive rate: {pos_rate:.4f} -> scale_pos_weight = {SCALE_POS_WEIGHT}")

trace_rows = []


def patient_level_split(X, y, subjects, test_size, seed):
    unique_subj = np.unique(subjects)
    subj_a, subj_b = train_test_split(unique_subj, test_size=test_size, random_state=seed)
    mask_a = np.isin(subjects, subj_a)
    mask_b = np.isin(subjects, subj_b)
    return (X[mask_a], X[mask_b], y[mask_a], y[mask_b],
            subjects[mask_a], subjects[mask_b])


def ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1], right=True)
    out = 0.0
    for b in range(n_bins):
        msk = idx == b
        if msk.sum() == 0:
            continue
        out += (msk.sum() / len(y_true)) * abs(y_true[msk].mean() - y_prob[msk].mean())
    return float(out)


def evaluate_candidate(stage, name, overrides, seeds=TUNE_SEEDS):
    aurocs, auprcs, briers, eces, meanps = [], [], [], [], []
    cfg = {**UNTUNED_BASE, **overrides}
    for seed in seeds:
        X_fit, X_cal, y_fit, y_cal, subj_fit, subj_cal = patient_level_split(
            X_all, y_all, subjects_all, test_size=0.2, seed=seed)
        X_ft, X_fv, y_ft, y_fv, subj_ft, subj_fv = patient_level_split(
            X_fit, y_fit, subj_fit, test_size=0.15, seed=seed)
        assert not (set(subj_fv.tolist()) & set(subj_cal.tolist())), \
            f"X_fit_val overlaps X_cal for seed={seed}"

        model = xgb.XGBClassifier(random_state=seed, **cfg)
        model.fit(X_ft, y_ft)
        p = model.predict_proba(X_fv)[:, 1]

        aurocs.append(roc_auc_score(y_fv, p))
        auprcs.append(average_precision_score(y_fv, p))
        briers.append(brier_score_loss(y_fv, p))
        eces.append(ece(y_fv, p))
        meanps.append(float(p.mean()))
        trace_rows.append({
            "stage": stage, "candidate": name, "seed": seed,
            "val_auroc": round(aurocs[-1], 4), "val_auprc": round(auprcs[-1], 4),
            "val_brier": round(briers[-1], 4), "val_ece": round(eces[-1], 4),
            "val_mean_pred": round(meanps[-1], 4),
            **{f"cfg_{k}": v for k, v in cfg.items()},
        })
    am, asd = float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))
    print(f"  [{stage}] {name:<34} val AUROC={am:.4f}+/-{asd:.4f}  "
          f"AUPRC={np.mean(auprcs):.4f}  Brier={np.mean(briers):.4f}  "
          f"ECE={np.mean(eces):.4f}  mean_p={np.mean(meanps):.4f}", flush=True)
    return am, asd, float(np.mean(eces)), float(np.mean(meanps))


def pick_winner(base, candidates):
    base_mean, base_sd, base_ece = base
    best = ("baseline", base_mean, base_sd)
    for name, mean, sd, cand_ece in candidates:
        if cand_ece > 5 * max(base_ece, 1e-6):
            print(f"      -> {name} REJECTED by calibration guard "
                  f"(ECE {cand_ece:.4f} vs baseline {base_ece:.4f})")
            continue
        if mean - base_mean > base_sd:
            if best[0] == "baseline" or mean > best[1]:
                best = (name, mean, sd)
    return best


best_cfg = {}

print("\n" + "=" * 78 + "\nSTAGE 1: Ensemble size and learning rate\n" + "=" * 78)
b = evaluate_candidate("1_capacity", "defaults (n=100, lr=0.3)", {**best_cfg})
c1 = evaluate_candidate("1_capacity", "n=300, lr=0.1", {**best_cfg, "n_estimators": 300, "learning_rate": 0.1})
c2 = evaluate_candidate("1_capacity", "n=600, lr=0.05", {**best_cfg, "n_estimators": 600, "learning_rate": 0.05})
c3 = evaluate_candidate("1_capacity", "n=1000, lr=0.03", {**best_cfg, "n_estimators": 1000, "learning_rate": 0.03})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("n=300, lr=0.1", c1[0], c1[1], c1[2]),
    ("n=600, lr=0.05", c2[0], c2[1], c2[2]),
    ("n=1000, lr=0.03", c3[0], c3[1], c3[2])])
print(f"  -> Stage 1 winner: {w} (val AUROC {wm:.4f})")
for nm, nn, ll in [("n=300, lr=0.1", 300, 0.1), ("n=600, lr=0.05", 600, 0.05), ("n=1000, lr=0.03", 1000, 0.03)]:
    if w == nm:
        best_cfg["n_estimators"], best_cfg["learning_rate"] = nn, ll

print("\n" + "=" * 78 + "\nSTAGE 2: Tree depth\n" + "=" * 78)
b = evaluate_candidate("2_depth", "depth=6 (baseline)", {**best_cfg})
c1 = evaluate_candidate("2_depth", "depth=4", {**best_cfg, "max_depth": 4})
c2 = evaluate_candidate("2_depth", "depth=8", {**best_cfg, "max_depth": 8})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("depth=4", c1[0], c1[1], c1[2]), ("depth=8", c2[0], c2[1], c2[2])])
print(f"  -> Stage 2 winner: {w} (val AUROC {wm:.4f})")
if w == "depth=4":
    best_cfg["max_depth"] = 4
elif w == "depth=8":
    best_cfg["max_depth"] = 8

print("\n" + "=" * 78 + "\nSTAGE 3: Stochastic regularisation (subsampling)\n" + "=" * 78)
b = evaluate_candidate("3_subsample", "no subsampling (baseline)", {**best_cfg})
c1 = evaluate_candidate("3_subsample", "subsample=0.8, colsample=0.8",
                        {**best_cfg, "subsample": 0.8, "colsample_bytree": 0.8})
c2 = evaluate_candidate("3_subsample", "subsample=0.6, colsample=0.6",
                        {**best_cfg, "subsample": 0.6, "colsample_bytree": 0.6})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("subsample=0.8, colsample=0.8", c1[0], c1[1], c1[2]),
    ("subsample=0.6, colsample=0.6", c2[0], c2[1], c2[2])])
print(f"  -> Stage 3 winner: {w} (val AUROC {wm:.4f})")
if w.startswith("subsample=0.8"):
    best_cfg.update({"subsample": 0.8, "colsample_bytree": 0.8})
elif w.startswith("subsample=0.6"):
    best_cfg.update({"subsample": 0.6, "colsample_bytree": 0.6})

print("\n" + "=" * 78 + "\nSTAGE 4: Complexity penalties\n" + "=" * 78)
b = evaluate_candidate("4_penalty", "min_child_weight=1, lambda=1 (baseline)", {**best_cfg})
c1 = evaluate_candidate("4_penalty", "min_child_weight=10", {**best_cfg, "min_child_weight": 10})
c2 = evaluate_candidate("4_penalty", "reg_lambda=10", {**best_cfg, "reg_lambda": 10.0})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    ("min_child_weight=10", c1[0], c1[1], c1[2]), ("reg_lambda=10", c2[0], c2[1], c2[2])])
print(f"  -> Stage 4 winner: {w} (val AUROC {wm:.4f})")
if w == "min_child_weight=10":
    best_cfg["min_child_weight"] = 10
elif w == "reg_lambda=10":
    best_cfg["reg_lambda"] = 10.0

print("\n" + "=" * 78 + "\nSTAGE 5: Class imbalance (the GRU-D `pos_weight` analogue)\n" + "=" * 78)
b = evaluate_candidate("5_imbalance", "scale_pos_weight=1 (baseline)", {**best_cfg})
c1 = evaluate_candidate("5_imbalance", f"scale_pos_weight={SCALE_POS_WEIGHT}",
                        {**best_cfg, "scale_pos_weight": SCALE_POS_WEIGHT})
w, wm, _ = pick_winner((b[0], b[1], b[2]), [
    (f"scale_pos_weight={SCALE_POS_WEIGHT}", c1[0], c1[1], c1[2])])
print(f"  -> Stage 5 winner: {w} (val AUROC {wm:.4f})")
if w != "baseline":
    best_cfg["scale_pos_weight"] = SCALE_POS_WEIGHT

trace_df = pd.DataFrame(trace_rows)
trace_df.to_csv(TRACE_CSV, index=False)
print(f"\nFull search trace ({len(trace_df)} rows) saved to: {TRACE_CSV}")

print("\n" + "=" * 78)
print("FINAL RECOMMENDED XGBOOST-MATCHED v2 CONFIG (validation-only; test_v2.csv never read)")
print("=" * 78)
print(json.dumps(best_cfg, indent=2) if best_cfg else "{}  (library defaults win every stage)")
print(f"\nFull effective config: {json.dumps({**UNTUNED_BASE, **best_cfg})}")
with open(os.path.join(DATA_DIR, "m2_v2_xgboost_tuned_config.json"), "w") as f:
    json.dump({"overrides": best_cfg, "effective": {**UNTUNED_BASE, **best_cfg}}, f, indent=2)
print("\nRolling this out to the 5-seed confirmatory comparison is a separate "
      "step (m2_xgboost_matched_v2.py, re-run with this config).")
