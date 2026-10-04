"""
M2 -- Patch: add absolute per-timestep elapsed hours (t_hours) to the
irregular-sequence npz files (Phase 5 prerequisite for mTAN).
==============================================================================
m2_build_irregular_cohort.py's per-variable `delta_hours` (Che et al. 2018
Eq 8-9: time since THIS variable's own last observation, accumulating
across consecutive misses) is what GRU-D needs, but mTAN
(enc_mtan_classif, src/models.py) needs a different, simpler quantity: the
single absolute elapsed time since triage (t=0) at each TIMESTEP, shared
across all 7 variables observed at that timestep -- not recoverable from
delta_hours alone once a variable has ever been missing. Cheap to
recompute directly (one pass over the same per-stay sequences,
independent of vitals cleaning/values), so this is a small additive patch
rather than a rebuild of the (already expensive) main pipeline.

Adds `t_hours` (N, T_MAX_TOTAL) to each existing
data/m2_irregular_sequences_{train_v2,test_v2}.npz, by merging (numpy
cannot append to an npz in place) -- every existing array is re-saved
unchanged alongside the new one, so no other consumer's data location or
schema needs updating.
"""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MIMIC_ED_PATH = os.path.join(SCRIPT_DIR, "..", "mimic4_full", "ed")

sys.path.insert(0, SCRIPT_DIR)
from m2_build_irregular_cohort import T_MAX_ED, T_MAX_TOTAL


def _prep_stay_times(vitalsign_df, t_max):
    """charttime-only counterpart of m2_build_irregular_cohort._prep_stay_arrays
    -- t_hours needs only the timestamps, not the vitals values, so this
    skips the vitals-column requirement merge_vitalsign_sequence_on_edstay
    otherwise imposes."""
    df = vitalsign_df.copy()
    df["charttime"] = pd.to_datetime(df["charttime"])
    df = df.sort_values(["stay_id", "charttime"])
    out = {}
    for sid, g in df.groupby("stay_id"):
        times_ns = g["charttime"].tail(t_max).values.astype("datetime64[ns]").astype(np.int64)
        out[sid] = times_ns
    return out


def compute_t_hours(df, stay_times):
    N = len(df)
    t_hours = np.zeros((N, T_MAX_TOTAL), dtype=np.float32)  # t=0 (triage) is always 0
    intime_ns = pd.to_datetime(df["intime"], errors="coerce").values.astype("datetime64[ns]").astype(np.int64)
    stay_ids = df["stay_id"].values
    for i in range(N):
        times_ns = stay_times.get(stay_ids[i])
        if times_ns is None or len(times_ns) == 0:
            continue
        k = min(len(times_ns), T_MAX_ED)
        t_hours[i, 1:1 + k] = (times_ns[:k].astype(np.float64) - intime_ns[i]) / 3.6e12
        if i % 50000 == 0:
            print(f"  ... {i}/{N}", flush=True)
    return t_hours


def main():
    vitalsign_raw = pd.read_csv(os.path.join(MIMIC_ED_PATH, "vitalsign.csv"), usecols=["stay_id", "charttime"])

    for split_name in ("train_v2", "test_v2"):
        print(f"\n{split_name}:")
        df = pd.read_csv(os.path.join(DATA_DIR, f"{split_name}.csv"), usecols=["stay_id", "intime"])
        combined_ids = df["stay_id"].unique()
        vs = vitalsign_raw[vitalsign_raw["stay_id"].isin(combined_ids)]
        stay_times = _prep_stay_times(vs, T_MAX_ED)

        t_hours = compute_t_hours(df, stay_times)

        npz_path = os.path.join(DATA_DIR, f"m2_irregular_sequences_{split_name}.npz")
        with np.load(npz_path) as z:
            existing = {k: z[k] for k in z.files}
        assert len(existing["values"]) == len(t_hours), "row count mismatch -- STOP"
        existing["t_hours"] = t_hours
        np.savez_compressed(npz_path, **existing)
        print(f"Patched {npz_path} with t_hours, shape {t_hours.shape}")


if __name__ == "__main__":
    main()
