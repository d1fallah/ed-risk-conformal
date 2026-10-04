"""
Guide Sec 18: "Paired prediction artifact: predictions_all_models.csv
containing patient_id/encounter_id + ground truth + every model's
predicted probability for paired analysis." Pivots the long-format
data/m2_hybrid_predictions.csv (one row per encounter x model x seed) into
one row per (subject_id, stay_id, seed) with one probability column per
model -- the shape needed for the patient-cluster bootstrap's paired
AUROC comparisons (Phase M).

Runs every integrity check Sec 18's "Prediction Artifact Integrity Checks"
lists, printing pass/fail for each rather than assuming success -- per the
project's golden rule, a claimed deliverable isn't complete until verified
against real output.
"""
import os, sys, hashlib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_PATH = os.path.join(DATA_DIR, "predictions_all_models.csv")

MODELS = ["Temporal-Only GRU-D", "Frozen-Temporal Head Control", "Static-Only MLP",
          "Hybrid GRU-D", "XGBoost-Matched"]
MODEL_COL = {m: "prob_" + m.lower().replace(" ", "_").replace("-", "_") for m in MODELS}

long_df = pd.read_csv(os.path.join(DATA_DIR, "m2_hybrid_predictions.csv"))
print(f"Loaded {len(long_df)} long-format rows across {long_df['model'].nunique()} models.")

# Ground truth must be identical across models for the same (stay_id, seed)
# -- verify BEFORE pivoting, since a silent mismatch would corrupt the
# paired comparison this file exists to support.
gt_check = long_df.groupby(["stay_id", "seed"])["y_true"].nunique()
assert (gt_check == 1).all(), f"y_true disagrees across models for {int((gt_check > 1).sum())} (stay_id,seed) pairs"
print("[PASS] Identical ground truth across all models, for every (stay_id, seed).")

wide = long_df.pivot_table(index=["subject_id", "stay_id", "seed"], columns="model",
                            values=["y_true", "y_prob"], aggfunc="first")
# One y_true per row (already verified identical across models above).
y_true_col = wide["y_true"].bfill(axis=1).iloc[:, 0]
probs = wide["y_prob"]
probs.columns = [MODEL_COL[m] for m in probs.columns]
out = pd.concat([y_true_col.rename("y_true"), probs], axis=1).reset_index()
out = out[["subject_id", "stay_id", "seed", "y_true"] + [MODEL_COL[m] for m in MODELS]]

# ── Sec 18 "Prediction Artifact Integrity Checks" -- verified, not assumed ──
checks = []

checks.append(("One row per encounter (per seed)",
                len(out) == out.groupby("seed").size().sum()))

dup = out.duplicated(subset=["subject_id", "stay_id", "seed"]).sum()
checks.append(("Unique (subject_id, stay_id, seed) triples", dup == 0))

checks.append(("Identical ground truth across all models",
                long_df.groupby(["stay_id", "seed"])["y_true"].nunique().eq(1).all()))

n_test_stays = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), usecols=["stay_id"])["stay_id"].nunique()
identical_cohort = all(
    set(out.loc[out["seed"] == s, "stay_id"]) == set(out.loc[out["seed"] == out["seed"].iloc[0], "stay_id"])
    for s in out["seed"].unique()
) and out["stay_id"].nunique() == n_test_stays
checks.append(("Identical test cohort across all models/seeds", identical_cohort))

n_missing = out[[MODEL_COL[m] for m in MODELS]].isna().sum().sum()
checks.append(("No missing prediction values", n_missing == 0))

in_range = out[[MODEL_COL[m] for m in MODELS]].apply(lambda c: c.between(0, 1).all()).all()
checks.append(("All predictions in range [0, 1]", bool(in_range)))

print(f"\n{'='*60}\nIntegrity checks\n{'='*60}")
all_passed = True
for name, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    all_passed = all_passed and passed
if not all_passed:
    raise AssertionError("One or more predictions_all_models.csv integrity checks failed -- see above.")

out.to_csv(OUT_PATH, index=False)
row_count = len(out)
with open(OUT_PATH, "rb") as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
print(f"\nWrote {OUT_PATH}")
print(f"Exact row count: {row_count}")
print(f"SHA-256: {file_hash}")

with open(os.path.join(DATA_DIR, "predictions_all_models_integrity.json"), "w") as f:
    import json
    json.dump({"row_count": row_count, "sha256": file_hash,
               "checks": [{"name": n, "passed": bool(p)} for n, p in checks]}, f, indent=2)
