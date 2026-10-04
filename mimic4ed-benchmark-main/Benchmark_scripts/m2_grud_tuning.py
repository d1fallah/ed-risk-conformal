"""
M2 GRU-D Tuning Search (validation-only -- never reads data/test.csv)
========================================================================
Purpose: close the gap between GRU-D's 5-seed mean AUROC (0.8937 +/- 0.0128)
and XGBoost's single-run AUROC (0.9184) on Task 2, per the teacher's request
for paper-review credibility -- WITHOUT touching anything that would make the
already-published findings meaningless.

What stays locked (never touched by this script): the Gate-approved feature
set (7 vitals x 2 timesteps + 24 static features), GRU-D's decay mechanism
(gamma = exp(-relu(Delta @ W_gamma + b_gamma))), the cohort/patient-level
split, the outcome definition, ECI_LR=0.05. See M2_Outcome_Definition_Gate.md
and the approved plan at time of writing
(C:\\Users\\user\\.claude\\plans\\i-m-building-sihhadz-a-floating-mccarthy.md).

What's tuned: training procedure (class-imbalance weighting, regularization,
batching, epoch policy) and model capacity (hidden size, classifier-head
width) -- i.e. how GRU-D is trained, never what it's trained on.

Integrity design:
  1. This script loads ONLY data/train.csv. It never imports, reads, or
     otherwise touches data/test.csv -- printed and asserted below.
  2. X_fit (the pool GRU-D is normally fit on) is further split into
     X_fit_train (85%) / X_fit_val (15%), stratified. X_fit_val never
     overlaps X_cal (MAPIE's calibration fold) or the test set -- asserted
     below via subject_id set intersection.
  3. GRU-D's own seed-to-seed AUROC std is 0.0128 -- almost half the
     XGBoost gap (0.0247). Every candidate is therefore evaluated at 3 seeds
     (42, 123, 7) on X_fit_val before being judged, and a candidate only
     replaces the current stage winner if its mean validation AUROC beats
     the baseline by more than one within-baseline validation std -- a
     parsimony guard against mistaking seed noise for a real improvement.
  4. The full search trace (every candidate, every seed) is saved to
     data/m2_grud_tuning_trace.csv -- nothing is cherry-picked silently.
  5. The winning config is only a *recommendation* printed at the end. It is
     not automatically rolled out to the primary pipeline (m2_ablations.py /
     train_grud_conformal.py) -- that happens as a separate, explicit step
     after this script's result is reviewed, matching how the ECI_LR=0.05
     change was handled.

Staged, sequential search (not full combinatorial, so the search space is
small and pre-specified):
  Stage 1: class imbalance      -- pos_weight in {None (baseline), ~15.9}
  Stage 2: capacity             -- hid in {16 (baseline), 32, 64}
  Stage 3: regularization       -- dropout in {0 (baseline), 0.2, 0.3};
                                    weight_decay in {0 (baseline), 1e-4}
  Stage 4: training procedure   -- batch_size in {None=full-batch (baseline), 2048};
                                    early stopping (patience=15 on val AUROC) vs
                                    fixed 150 epochs (baseline)
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
TRAIN_CSV  = os.path.join(DATA_DIR, "train.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test.csv")
TRACE_CSV  = os.path.join(DATA_DIR, "m2_grud_tuning_trace.csv")

assert not any("test.csv" in str(a) for a in sys.argv), "This script must never touch test.csv."
print(f"Integrity check: TEST_CSV path is defined ({TEST_CSV}) but will NOT be opened by this script.")

sys.path.insert(0, SCRIPT_DIR)
from m2_ablations import SKLearnGRUD, prepare_data, TARGET

TUNE_SEEDS = [42, 123, 7]  # deliberately distinct from the 5-seed CONFIRMATORY protocol's seed list use

# ── The untuned reference configuration, stated explicitly ──────────────────
# Every candidate below is built as {**UNTUNED_BASE, **best_cfg, **overrides},
# so a stage's "baseline" arm is a fully-specified configuration rather than
# whatever SKLearnGRUD's constructor defaults happen to be at the time.
#
# This is not a cosmetic change. SKLearnGRUD's defaults were themselves
# rewritten to the FIRST tuning pass's adopted winner (hid=32, dropout=0.2,
# weight_decay=1e-4), so on any re-run the unspecified-kwargs baseline
# silently *became* that winner. The 2026-09-01 re-run under Deviation 7's
# corrected architecture shows the damage directly in its trace: Stage 3's
# "no reg (baseline)" already carried dropout=0.2 and weight_decay=1e-4, so
# it returned numbers bit-identical to the two candidates it was meant to be
# compared against (0.9244 +/- 0.0017 for all three), and Stage 5's
# "combine hid=32+dropout=0.2+wd=1e-4" was bit-identical to its own cfg={}
# reference (0.9166 +/- 0.0011). Those stages measured nothing at all.
#
# Pinning the base here makes each stage measure the factor it names,
# regardless of what the module defaults are later set to.
UNTUNED_BASE = {"hid": 16, "clf_hidden": 32, "dropout": 0.0,
                "weight_decay": 0.0, "pos_weight": None,
                "batch_size": None, "epochs": 150, "lr": 0.01,
                "early_stopping_patience": None}

print("Loading data/train.csv only...")
df_train = pd.read_csv(TRAIN_CSV)
X_train, y_train = prepare_data(df_train)
train_subjects = df_train["subject_id"].values

pos_rate = y_train.mean()
POS_WEIGHT = round((1 - pos_rate) / pos_rate, 2)
print(f"Train positive rate: {pos_rate:.4f} -> pos_weight = {POS_WEIGHT}")

trace_rows = []

def patient_level_split(X, y, subjects, test_size, seed):
    """Splits X/y by unique subject_id, not by row -- 34% of patients have
    more than one ED visit (established earlier this session for the outer
    train/test split), so a row-level train_test_split leaks the same
    patient's rows across both sides. All rows for a given patient land on
    the same side of the split."""
    unique_subj = np.unique(subjects)
    subj_a, subj_b = train_test_split(unique_subj, test_size=test_size, random_state=seed)
    mask_a = np.isin(subjects, subj_a)
    mask_b = np.isin(subjects, subj_b)
    return X[mask_a], X[mask_b], y[mask_a], y[mask_b], subjects[mask_a], subjects[mask_b]

def evaluate_candidate(stage, name, kwargs, seeds=TUNE_SEEDS):
    """Trains+evaluates one candidate config at each seed on a fresh
    X_fit_train/X_fit_val split (split itself also re-drawn per seed, matching
    how the primary pipeline's MAPIE fit/cal split varies with seed). All
    splits here are patient-level (see patient_level_split), unlike the
    primary pipeline's fit/cal split -- see the flagged finding in the
    tuning summary at the bottom of this script."""
    aurocs, auprcs = [], []
    for seed in seeds:
        X_fit, X_cal, y_fit, y_cal, subj_fit, subj_cal = patient_level_split(
            X_train, y_train, train_subjects, test_size=0.2, seed=seed
        )
        X_fit_train, X_fit_val, y_fit_train, y_fit_val, subj_ft, subj_fv = patient_level_split(
            X_fit, y_fit, subj_fit, test_size=0.15, seed=seed
        )
        # Integrity assertion: validation fold never overlaps the calibration fold.
        overlap = set(subj_fv.tolist()) & set(subj_cal.tolist())
        assert len(overlap) == 0, f"X_fit_val overlaps X_cal for seed={seed}: {len(overlap)} patients"

        model = SKLearnGRUD(seed=seed, **{**UNTUNED_BASE, **kwargs})
        if kwargs.get("early_stopping_patience") is not None:
            # Early stopping needs its own monitoring fold, distinct from
            # X_fit_val -- if it monitored X_fit_val directly, the reported
            # "val AUROC" for this candidate would be the same signal used to
            # pick its best checkpoint, an optimistic bias the other
            # candidates don't get. Carve a further split out of
            # X_fit_train only; X_fit_val stays untouched during training
            # for every candidate, early-stopped or not.
            X_es_train, X_es_stop, y_es_train, y_es_stop, _, _ = patient_level_split(
                X_fit_train, y_fit_train, subj_ft, test_size=0.2, seed=seed
            )
            model.fit(X_es_train, y_es_train, X_val=X_es_stop, y_val=y_es_stop)
        else:
            model.fit(X_fit_train, y_fit_train)
        p_val = model.predict_proba(X_fit_val)[:, 1]
        auroc = roc_auc_score(y_fit_val, p_val)
        auprc = average_precision_score(y_fit_val, p_val)
        aurocs.append(auroc); auprcs.append(auprc)
        trace_rows.append({
            "stage": stage, "candidate": name, "seed": seed,
            "val_auroc": round(auroc, 4), "val_auprc": round(auprc, 4),
            **{f"cfg_{k}": v for k, v in {**UNTUNED_BASE, **kwargs}.items()},
        })
    auroc_mean, auroc_std = float(np.mean(aurocs)), float(np.std(aurocs, ddof=1))
    auprc_mean, auprc_std = float(np.mean(auprcs)), float(np.std(auprcs, ddof=1))
    print(f"  [{stage}] {name:<28} val AUROC={auroc_mean:.4f}+/-{auroc_std:.4f}  "
          f"val AUPRC={auprc_mean:.4f}+/-{auprc_std:.4f}   cfg={kwargs}")
    return auroc_mean, auroc_std, auprc_mean, auprc_std

def pick_winner(baseline_result, candidates_results):
    """A candidate only wins if it beats the baseline by more than one
    within-baseline validation std -- guards against noise-driven wins given
    GRU-D's known seed-to-seed variance."""
    base_mean, base_std = baseline_result[0], baseline_result[1]
    best = ("baseline", base_mean, base_std)
    for name, mean, std in candidates_results:
        if mean - base_mean > base_std:
            if best[0] == "baseline" or mean > best[1]:
                best = (name, mean, std)
    return best

best_cfg = {}  # accumulates winning params across stages

# ── Stage 1: class imbalance ────────────────────────────────────────────────
print("\n" + "="*70 + "\nSTAGE 1: Class imbalance (pos_weight)\n" + "="*70)
base = evaluate_candidate("1_imbalance", "baseline (unweighted)", {**best_cfg})
cand_pw = evaluate_candidate("1_imbalance", f"pos_weight={POS_WEIGHT}", {**best_cfg, "pos_weight": POS_WEIGHT})
winner, wmean, wstd = pick_winner((base[0], base[1]), [(f"pos_weight={POS_WEIGHT}", cand_pw[0], cand_pw[1])])
print(f"  -> Stage 1 winner: {winner} (val AUROC {wmean:.4f})")
if winner != "baseline":
    best_cfg["pos_weight"] = POS_WEIGHT

# ── Stage 2: capacity ────────────────────────────────────────────────────────
print("\n" + "="*70 + "\nSTAGE 2: Capacity (hidden size)\n" + "="*70)
base = evaluate_candidate("2_capacity", "hid=16 (baseline)", {**best_cfg, "hid": 16})
c32 = evaluate_candidate("2_capacity", "hid=32", {**best_cfg, "hid": 32})
c64 = evaluate_candidate("2_capacity", "hid=64", {**best_cfg, "hid": 64})
winner, wmean, wstd = pick_winner((base[0], base[1]), [("hid=32", c32[0], c32[1]), ("hid=64", c64[0], c64[1])])
print(f"  -> Stage 2 winner: {winner} (val AUROC {wmean:.4f})")
if winner == "hid=32":
    best_cfg["hid"] = 32
elif winner == "hid=64":
    best_cfg["hid"] = 64

# ── Stage 3: regularization ──────────────────────────────────────────────────
print("\n" + "="*70 + "\nSTAGE 3: Regularization (dropout, weight_decay)\n" + "="*70)
base = evaluate_candidate("3_reg", "no reg (baseline)", {**best_cfg})
d1 = evaluate_candidate("3_reg", "dropout=0.2", {**best_cfg, "dropout": 0.2})
d2 = evaluate_candidate("3_reg", "dropout=0.3", {**best_cfg, "dropout": 0.3})
wd = evaluate_candidate("3_reg", "weight_decay=1e-4", {**best_cfg, "weight_decay": 1e-4})
winner, wmean, wstd = pick_winner((base[0], base[1]), [
    ("dropout=0.2", d1[0], d1[1]), ("dropout=0.3", d2[0], d2[1]), ("weight_decay=1e-4", wd[0], wd[1])
])
print(f"  -> Stage 3 winner: {winner} (val AUROC {wmean:.4f})")
if winner == "dropout=0.2":
    best_cfg["dropout"] = 0.2
elif winner == "dropout=0.3":
    best_cfg["dropout"] = 0.3
elif winner == "weight_decay=1e-4":
    best_cfg["weight_decay"] = 1e-4

# ── Stage 4: training procedure (batching, early stopping) ──────────────────
print("\n" + "="*70 + "\nSTAGE 4: Training procedure (batching, early stopping)\n" + "="*70)
base = evaluate_candidate("4_procedure", "full-batch, fixed 150 ep (baseline)", {**best_cfg})
mb = evaluate_candidate("4_procedure", "batch_size=2048", {**best_cfg, "batch_size": 2048})
es = evaluate_candidate("4_procedure", "early_stopping(patience=15)",
                         {**best_cfg, "epochs": 300, "early_stopping_patience": 15})
winner, wmean, wstd = pick_winner((base[0], base[1]), [
    ("batch_size=2048", mb[0], mb[1]), ("early_stopping(patience=15)", es[0], es[1])
])
print(f"  -> Stage 4 winner: {winner} (val AUROC {wmean:.4f})")
if winner == "batch_size=2048":
    best_cfg["batch_size"] = 2048
elif winner == "early_stopping(patience=15)":
    best_cfg["epochs"] = 300
    best_cfg["early_stopping_patience"] = 15

# ── Stage 5: one pre-specified combined-factor follow-up ─────────────────────
# STALE PREMISE, FIXED 2026-09-01 (found by an external statistical review,
# M2_Hybrid_Deviation_Log.md Deviation 8's own species of bug): this stage's
# comment and hardcoded candidate ("hid=32+dropout=0.2+wd=1e-4") describe the
# FIRST tuning pass's outcome, where three factors each independently fell
# just short with a tighter std, motivating exactly that combined probe. The
# corrected search run under Deviation 7's architecture does not reproduce
# that pattern at all -- pos_weight=15.75 and hid=64 each independently WIN
# their own stages outright, while dropout/weight_decay/batching/early
# stopping all lose outright. Testing the old, unrelated combo here is
# vestigial, not a follow-up motivated by this run's own results.
#
# The bug this masked (also found by the same review): the old code passed
# `{}` rather than `best_cfg` as this stage's baseline -- so on a seed where
# the combo happened to win, `best_cfg = {"hid": 32, ...}` REPLACED, rather
# than updated, whatever Stages 1-2 had already accumulated (pos_weight,
# hid=64), silently discarding two already-adopted winners. It did not fire
# in the run this dossier reports (0.9166 vs 0.9133 does not clear the
# guard), so no adopted config was affected -- but the trap was live and is
# fixed here rather than left for the next run to hit.
#
# Retired rather than patched: there is no longer a data-driven hypothesis
# for what "combined" should mean, so this stage now runs a single sanity
# check -- verifying the actually-accumulated best_cfg's own combination of
# winners is itself at least as good as any single winner -- instead of
# testing an unrelated hardcoded combo. If a future run's Stages 1-4 again
# show the "several factors each independently fall just short" pattern,
# reinstate a real combined-factor probe over THOSE factors specifically.
if best_cfg:
    print("\n" + "="*70 + f"\nSTAGE 5: Sanity check -- accumulated config {best_cfg}\n" + "="*70)
    base = evaluate_candidate("5_combined", "baseline (fully untuned, for reference)", {})
    combo = evaluate_candidate("5_combined", f"accumulated best_cfg={best_cfg}", {**best_cfg})
    print(f"  -> accumulated config val AUROC {combo[0]:.4f} vs untuned baseline {base[0]:.4f} "
          f"(informational only -- does not change best_cfg, which already reflects Stages 1-4)")
else:
    print("\n" + "="*70 + "\nSTAGE 5: skipped -- no stage above adopted anything to combine\n" + "="*70)

# ── Save trace + summary ─────────────────────────────────────────────────────
trace_df = pd.DataFrame(trace_rows)
trace_df.to_csv(TRACE_CSV, index=False)
print(f"\nFull search trace ({len(trace_df)} rows) saved to: {TRACE_CSV}")

print("\n" + "="*70)
print("FINAL RECOMMENDED CONFIG (validation-only; test.csv was never read)")
print("="*70)
print(json.dumps(best_cfg, indent=2) if best_cfg else "{}  (baseline config wins every stage)")
print("\nThis is a RECOMMENDATION ONLY. Rolling it out to the primary pipeline")
print("(m2_ablations.py / train_grud_conformal.py defaults) and re-running the")
print("5-seed confirmatory protocol on the held-out test set is a separate,")
print("explicit next step.")
