"""
M2 -- Recompute score_NEWS2 in train.csv/test.csv after Deviation 10's fix
==========================================================================
add_score_NEWS2() (helpers.py) previously implemented only 4 of NEWS2's
parameters (respiratory rate, temperature, SBP, heart rate); SpO2 was absent
despite `triage_o2sat` being present and cleaned in this same extract. This
script re-applies the corrected function to the already-frozen train.csv/
test.csv and overwrites just the `score_NEWS2` column in place -- it does
NOT re-run the full extraction/imputation pipeline (data_general_processing.py),
so no other column, and no split, is touched.

Per this project's standing practice, the old column is archived before being
overwritten rather than silently discarded.

Output: data/score_NEWS2_pre_deviation10_{train,test}.csv (old values, keyed
by stay_id, for audit) and train.csv/test.csv with score_NEWS2 corrected.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
sys.path.insert(0, SCRIPT_DIR)

from helpers import add_score_NEWS2

for name in ("test.csv", "train.csv"):
    path = os.path.join(DATA_DIR, name)
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    df = pd.read_csv(path)

    old = df[["stay_id", "score_NEWS2"]].copy()
    old_path = os.path.join(DATA_DIR, f"score_NEWS2_pre_deviation10_{name}")
    old.to_csv(old_path, index=False)
    print(f"  archived old score_NEWS2 -> {old_path}")

    old_vals = df["score_NEWS2"].copy()
    add_score_NEWS2(df)
    changed = (df["score_NEWS2"] != old_vals).sum()
    print(f"  score_NEWS2 changed on {changed}/{len(df)} rows "
          f"({changed/len(df)*100:.1f}%)")
    print(f"  old: min={old_vals.min()} max={old_vals.max()} mean={old_vals.mean():.4f}")
    print(f"  new: min={df['score_NEWS2'].min()} max={df['score_NEWS2'].max()} "
          f"mean={df['score_NEWS2'].mean():.4f}")

    df.to_csv(path, index=False)
    print(f"  saved corrected {name}")

print("\nDone. Re-run run_news2_baseline.py to get the corrected AUROC/AUPRC.")
