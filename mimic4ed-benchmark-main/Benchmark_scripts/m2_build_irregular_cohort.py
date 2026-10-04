"""
M2 -- Irregular-Series Cohort Rebuild (Phase 2, session M2 Q1-blockers plan)
==============================================================================
Implements M2_Outcome_Definition_Gate.md v2 (signed off 2026-09-19 by Akram):
  Sec 3 -- real time-to-event horizon labels, replacing the ed_los_hours<=h
           LOS filter.
  Sec 4 -- the genuine irregular ed/vitalsign.csv per-stay sequence
           (T_max=10 most recent + triage), replacing the fixed T=2
           (triage + last) representation.
  Sec 7 -- chronological (rolling-origin) split promoted to PRIMARY; the
           existing random 80/20 split retained as a SECONDARY robustness
           arm, not deleted.

Reuses rather than reimplements: cohort filters and vitals cleaning from
data_general_processing.py's own table (imported, not duplicated, from
m2_hybrid_common.VITALS_VALID_RANGE, which itself asserts parity with that
file); the chronological-split machinery from m2_chronological_eci.py
(chronological_split, CUT_YEAR) and m2_hybrid_common.py
(add_temporal_axis/_load_real_time_lookup, the anchor_year_group real-time
recovery); the 4-way dev-pool partition PATTERN from
m2_hybrid_common.build_fixed_dev_split (generalized here to a time-ordered
partition on the primary split, since a chronological primary split should
hold out its MOST RECENT patients for calibration, matching
chronological_fit_cal's existing reasoning, rather than a random 4-way
slice of an already-past-only pool).

Archives ALL pre-rebuild artifacts to data/archive_pre_irregular/ before
writing anything new (done at the start of this session, see that
directory) -- this script writes NEW files (suffixed _v2 / prefixed
m2_irregular_) rather than overwriting train.csv/test.csv/master_dataset.csv,
so the T=2 pipeline's already-computed, now-historical 5-seed results
remain reproducible from the archived copies.

Outputs:
  data/train_v2.csv, data/test_v2.csv        -- primary (chronological) split,
                                                 same row schema as train.csv/
                                                 test.csv plus event_time and
                                                 y_1h/y_6h/y_12h/horizon_eligible.
  data/m2_v2_random_split_ids.json           -- secondary (random) split,
                                                 subject IDs + SHA-256, for
                                                 the robustness arm.
  data/m2_v2_dev_pool_split.json             -- 4-way time-ordered partition
                                                 of the chronological primary
                                                 train period.
  data/m2_irregular_sequences.npz            -- per-stay irregular vitals
                                                 tensors: values, obs_mask,
                                                 pad_mask, delta_hours (all
                                                 shape (N, 1+T_max, 7)),
                                                 keyed by row order in
                                                 train_v2.csv + test_v2.csv
                                                 (row index saved alongside).
"""
import os, sys, json, hashlib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MIMIC_ED_PATH = os.path.join(SCRIPT_DIR, "..", "mimic4_full", "ed")
MIMIC_HOSP_PATH = os.path.join(SCRIPT_DIR, "..", "mimic4_full", "hosp")

sys.path.insert(0, SCRIPT_DIR)
from helpers import (
    convert_temp_to_celcius, remove_outliers, merge_vitalsign_sequence_on_edstay,
    convert_str_to_float, read_triage_table,
)
from m2_hybrid_common import (
    VITALS_COLS, VITALS_VALID_RANGE, add_temporal_axis, _hash_ids,
)
from m2_chronological_eci import chronological_split, CUT_YEAR

T_MAX_ED = 10          # p90 of ED-vitalsign observations per stay (measured this session)
T_MAX_TOTAL = 1 + T_MAX_ED  # + triage at t=0
HORIZONS_H = [1, 6, 12]
DEV_SPLIT_FRACTIONS = [70 / 90, 10 / 90, 5 / 90, 5 / 90]  # same ratios as the T=2 pipeline
DEV_SPLIT_NAMES = ["training", "validation", "calibration_selection", "final_conformal_calibration"]


def build_horizon_labels(df):
    """Gate v2 Sec 3: event_time = min(first-direct-ICU-transfer intime,
    in-hospital deathtime); y_h = 1{0 <= event_time-origin <= h}, origin =
    intime (triage, T=0). Requires df to already carry `intime`,
    `intime_icu`, `outcome_icu_transfer_12h` (from master_dataset.csv) and
    `deathtime` (joined here from hosp/admissions.csv, since
    master_dataset.csv does not carry it)."""
    intime = pd.to_datetime(df["intime"], errors="coerce")
    intime_icu = pd.to_datetime(df["intime_icu"], errors="coerce")
    deathtime = pd.to_datetime(df["deathtime"], errors="coerce")

    # icustays.intime is only a genuine "event" for a DIRECT ED->ICU
    # transfer -- matching outcome_icu_transfer_12h's own definition
    # (helpers.py::add_outcome_icu_transfer), not any later ICU stay.
    icu_event_time = intime_icu.where(df["outcome_icu_transfer_12h"].astype(bool))
    event_time = pd.concat([icu_event_time, deathtime], axis=1).min(axis=1)

    hours_to_event = (event_time - intime).dt.total_seconds() / 3600.0
    has_event_time = event_time.notna()
    non_negative = hours_to_event >= 0
    excluded_no_event_time = df["outcome_critical"].astype(bool) & ~has_event_time
    excluded_negative = has_event_time & ~non_negative
    horizon_eligible = ~(excluded_no_event_time | excluded_negative)

    out = df.copy()
    out["event_time"] = event_time
    out["hours_to_event"] = hours_to_event
    out["horizon_eligible"] = horizon_eligible
    for h in HORIZONS_H:
        y_h = has_event_time & non_negative & (hours_to_event <= h)
        out[f"y_{h}h"] = y_h.astype(int)
        # a horizon label is only meaningful for horizon-eligible rows;
        # ineligible rows get NaN rather than a silently-wrong 0, so any
        # downstream mean()/AUROC on this column fails loudly if the
        # exclusion isn't applied first.
        out.loc[~horizon_eligible, f"y_{h}h"] = np.nan

    n_excl_no_et = int(excluded_no_event_time.sum())
    n_excl_neg = int(excluded_negative.sum())
    print(f"Horizon labels: {n_excl_no_et} critical encounters with no recoverable event_time, "
          f"{n_excl_neg} with event_time < intime -- both excluded from horizon-eligible analyses "
          f"({n_excl_no_et + n_excl_neg} total, gate v2 Sec 3/6).")
    for h in HORIZONS_H:
        n_pos = int(out.loc[horizon_eligible, f"y_{h}h"].sum())
        n_elig = int(horizon_eligible.sum())
        print(f"  h={h}h: {n_pos} positive / {n_elig} eligible ({n_pos/n_elig:.5f})")
    return out


def load_and_filter_master():
    print("Loading master_dataset.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"))
    print(f"  {len(df)} rows")
    df = df[df["age"] >= 18]
    df = df[df["triage_acuity"].notnull()]
    print(f"  {len(df)} rows after age>=18 and non-null triage_acuity (Gate v2 Sec 5, unchanged from v1)")

    # master_dataset.csv already carries `deathtime` from the original
    # extraction's own admissions join (helpers.py::read_admissions_table) --
    # verified directly (4,600 of 4,733 outcome_inhospital_mortality=True
    # rows have a non-null deathtime, matching this session's earlier
    # cross-check against hosp/admissions.csv exactly). No second join
    # needed; re-merging hosp/admissions.csv here would only create a
    # duplicate deathtime_x/deathtime_y column pair.
    assert "deathtime" in df.columns, "master_dataset.csv is missing deathtime -- re-run extract_master_dataset.py"

    df = convert_temp_to_celcius(df)
    df = remove_outliers(df, VITALS_VALID_RANGE)

    df = build_horizon_labels(df)
    return df


def build_dev_pool_split_v2(train_period_df):
    """4-way time-ordered partition of the chronological primary train
    period, generalizing chronological_fit_cal's 2-way (fit/cal) holdout
    -- and the T=2 pipeline's build_fixed_dev_split's random 4-way
    partition -- to a partition that respects time order: the MOST RECENT
    patients become final_conformal_calibration and calibration_selection
    (matching how a deployed system would calibrate on recent history
    before predicting forward), validation is the next-most-recent slice,
    and training is the oldest, largest slice. Same fractions as the T=2
    pipeline (70/90, 10/90, 5/90, 5/90 of the pool)."""
    first_seen = train_period_df.groupby("subject_id")["approx_real_year"].min().sort_values()
    ids_oldest_first = first_seen.index.values
    n = len(ids_oldest_first)
    n_train = int(round(n * DEV_SPLIT_FRACTIONS[0]))
    n_val = int(round(n * DEV_SPLIT_FRACTIONS[1]))
    n_cal_sel = int(round(n * DEV_SPLIT_FRACTIONS[2]))
    # final_conformal_calibration takes the remainder, so the 4 groups are
    # exhaustive by construction regardless of rounding.
    cut1 = n_train
    cut2 = n_train + n_val
    cut3 = n_train + n_val + n_cal_sel
    groups = {
        "training": ids_oldest_first[:cut1],
        "validation": ids_oldest_first[cut1:cut2],
        "calibration_selection": ids_oldest_first[cut2:cut3],
        "final_conformal_calibration": ids_oldest_first[cut3:],
    }
    result = {}
    for name in DEV_SPLIT_NAMES:
        ids_sorted = np.sort(groups[name])
        result[name] = {"subject_ids": ids_sorted, "sha256": _hash_ids(ids_sorted)}
    all_ids = np.concatenate([v["subject_ids"] for v in result.values()])
    assert len(all_ids) == n and len(np.unique(all_ids)) == n, "v2 dev-pool split not disjoint/exhaustive"
    return result


def _prep_stay_arrays(vitalsign_seq_by_stay):
    """One-time conversion of {stay_id: DataFrame} (pandas, slow per-row
    access) into {stay_id: (times_ns int64 array, vals float32 (k,7)
    array)} (plain numpy, fast per-row access) -- done once, N_stays
    times, rather than inside the O(N_rows) main loop below."""
    out = {}
    for sid, seq in vitalsign_seq_by_stay.items():
        if len(seq) == 0:
            continue
        times_ns = seq["charttime"].values.astype("datetime64[ns]").astype(np.int64)
        vals = seq[VITALS_COLS].to_numpy(dtype=np.float64)
        out[sid] = (times_ns, vals)
    return out


def build_irregular_sequences(df, triage_df, vitalsign_seq_by_stay):
    """Builds the (N, T_MAX_TOTAL, 7) irregular tensors for every row of
    df (in df's row order), Che et al. (2018) Eq 8-9 per-variable decay:
    delta_t^v = (time since t-1) if v observed at t-1, else
    delta_t^v = delta_{t-1}^v + (time since t-1) -- accumulates across
    consecutive missing observations of THAT variable specifically, not a
    single shared per-timestep delta.

    Performance note: this cohort is ~425k rows. A naive per-row loop
    using pandas .loc/.iloc for every triage and vitalsign lookup was
    measured to be impractically slow at this scale (pandas per-call
    overhead dominates). Triage vitals are therefore joined onto df ONCE,
    vectorized, before the loop (`_TRIAGE_COLS_NP` below); vitalsign
    per-stay sequences are pre-converted to plain numpy arrays once via
    `_prep_stay_arrays` rather than re-accessed as DataFrames inside the
    loop. The remaining per-row loop is over plain numpy scalars/arrays
    only -- no pandas objects are touched inside it.
    """
    N = len(df)
    values = np.full((N, T_MAX_TOTAL, 7), np.nan, dtype=np.float32)
    obs_mask = np.zeros((N, T_MAX_TOTAL, 7), dtype=np.float32)
    pad_mask = np.zeros((N, T_MAX_TOTAL), dtype=np.float32)
    delta_hours = np.zeros((N, T_MAX_TOTAL, 7), dtype=np.float32)
    pad_mask[:, 0] = 1.0  # triage is always present as a timestep, even if every value is missing

    # --- Vectorized triage join (once, not per-row) ---
    triage_cols = [f"triage_{v}" for v in VITALS_COLS]
    triage_small = triage_df[["stay_id"] + triage_cols].drop_duplicates(subset="stay_id")
    df_idx = df[["stay_id"]].merge(triage_small, on="stay_id", how="left")
    triage_vals = df_idx[triage_cols].to_numpy(dtype=np.float64)  # (N, 7)
    triage_obs = ~np.isnan(triage_vals)
    values[:, 0, :] = triage_vals
    obs_mask[:, 0, :] = triage_obs.astype(np.float32)

    # --- Pre-convert per-stay ED-vitalsign sequences to numpy once ---
    stay_arrays = _prep_stay_arrays(vitalsign_seq_by_stay)

    intime_ns = pd.to_datetime(df["intime"], errors="coerce").values.astype("datetime64[ns]").astype(np.int64)
    stay_ids = df["stay_id"].values

    n_truncated = 0
    n_zero_ed_obs = 0
    for i in range(N):
        sid = stay_ids[i]
        t0_ns = intime_ns[i]

        entry = stay_arrays.get(sid)
        if entry is None:
            n_zero_ed_obs += 1
            continue
        times_ns, vals = entry
        k = min(len(times_ns), T_MAX_ED)
        if len(times_ns) > T_MAX_ED:
            n_truncated += 1

        cur_hours_arr = (times_ns[:k].astype(np.float64) - t0_ns) / 3.6e12  # ns -> hours
        obs_k = ~np.isnan(vals[:k, :])
        vals_k = vals[:k, :]

        pad_mask[i, 1:1 + k] = 1.0
        values[i, 1:1 + k, :] = vals_k
        obs_mask[i, 1:1 + k, :] = obs_k.astype(np.float32)

        last_obs_hours = np.zeros(7, dtype=np.float64)
        have_seen = triage_obs[i].copy()
        last_obs_hours[have_seen] = 0.0  # triage, if observed for that var, is "last seen" at hour 0
        prev_t_hours = 0.0
        for j in range(k):
            t_idx = 1 + j
            cur_hours = cur_hours_arr[j]
            for v_idx in range(7):
                if obs_k[j, v_idx]:
                    gap_since_last = cur_hours - last_obs_hours[v_idx] if have_seen[v_idx] else cur_hours
                    delta_hours[i, t_idx, v_idx] = max(0.0, gap_since_last)
                    last_obs_hours[v_idx] = cur_hours
                    have_seen[v_idx] = True
                else:
                    gap = max(0.0, cur_hours - prev_t_hours)
                    delta_hours[i, t_idx, v_idx] = delta_hours[i, t_idx - 1, v_idx] + gap
            prev_t_hours = cur_hours

        if i % 50000 == 0:
            print(f"  ... {i}/{N} rows built", flush=True)

    print(f"Irregular sequences built: {N} rows, {n_zero_ed_obs} with zero ED-vitalsign rows "
          f"({n_zero_ed_obs/N:.4f}), {n_truncated} truncated to T_MAX_ED={T_MAX_ED} "
          f"({n_truncated/N:.4f}).")
    return values, obs_mask, pad_mask, delta_hours


def main():
    df = load_and_filter_master()
    df = add_temporal_axis(df)  # approx_real_year, for the chronological split

    # ---- Primary split: chronological, promoted per Gate v2 Sec 7 ----
    train_period, test_period, dropped = chronological_split(df)
    print(f"\nChronological (primary) split, cut at {CUT_YEAR}: "
          f"train {len(train_period)} rows / {train_period['subject_id'].nunique()} patients, "
          f"test {len(test_period)} rows / {test_period['subject_id'].nunique()} patients, "
          f"dropped {len(dropped)} rows / {dropped['subject_id'].nunique()} patients "
          f"(straddled the cut, assigned to train period, evaluation dropped).")

    # ---- Secondary split: random 80/20, retained as a robustness arm ----
    from sklearn.model_selection import train_test_split
    unique_subjects = df["subject_id"].unique()
    random_train_subjects, random_test_subjects = train_test_split(
        unique_subjects, train_size=0.8, random_state=42)
    with open(os.path.join(DATA_DIR, "m2_v2_random_split_ids.json"), "w") as f:
        json.dump({
            "train_subject_ids": {"subject_ids": np.sort(random_train_subjects).tolist(),
                                    "sha256": _hash_ids(random_train_subjects)},
            "test_subject_ids": {"subject_ids": np.sort(random_test_subjects).tolist(),
                                   "sha256": _hash_ids(random_test_subjects)},
        }, f, indent=2)
    print(f"Random (secondary) split persisted: {len(random_train_subjects)} train / "
          f"{len(random_test_subjects)} test patients.")

    # ---- 4-way dev-pool partition of the chronological train period ----
    dev_split = build_dev_pool_split_v2(train_period)
    with open(os.path.join(DATA_DIR, "m2_v2_dev_pool_split.json"), "w") as f:
        json.dump({k: {"subject_ids": v["subject_ids"].tolist(), "sha256": v["sha256"]}
                   for k, v in dev_split.items()}, f, indent=2)
    print("v2 dev-pool split (time-ordered, chronological primary train period): "
          + ", ".join(f"{k}={len(v['subject_ids'])}" for k, v in dev_split.items()))

    # ---- Write train_v2.csv / test_v2.csv ----
    train_period = train_period.copy()
    test_period = test_period.copy()
    train_period.to_csv(os.path.join(DATA_DIR, "train_v2.csv"), index=False)
    test_period.to_csv(os.path.join(DATA_DIR, "test_v2.csv"), index=False)
    print(f"Wrote data/train_v2.csv ({len(train_period)} rows), "
          f"data/test_v2.csv ({len(test_period)} rows).")

    # ---- Build irregular vitals sequences for train_v2 + test_v2 (in that row order) ----
    print("\nLoading raw triage.csv and ed/vitalsign.csv for sequence extraction...")
    # read_triage_table already: renames to the triage_* prefix convention
    # convert_temp_to_celcius/remove_outliers key off, AND applies
    # convert_str_to_float to triage_pain (raw pain values are sometimes
    # non-numeric strings, e.g. range expressions) -- reused as-is rather
    # than reimplemented.
    triage_raw = read_triage_table(os.path.join(MIMIC_ED_PATH, "triage.csv"))
    triage_raw = convert_temp_to_celcius(triage_raw)
    triage_raw = remove_outliers(triage_raw, VITALS_VALID_RANGE)

    vitalsign_raw = pd.read_csv(os.path.join(MIMIC_ED_PATH, "vitalsign.csv"))
    combined_ids = pd.concat([train_period["stay_id"], test_period["stay_id"]]).unique()
    vitalsign_raw = vitalsign_raw[vitalsign_raw["stay_id"].isin(combined_ids)].copy()
    vitalsign_raw["pain"] = vitalsign_raw["pain"].apply(convert_str_to_float).astype(float)
    # convert_temp_to_celcius / remove_outliers both key off a "prefix_vital"
    # column-naming convention (column.split('_')[1]) -- vitalsign.csv's raw
    # columns are bare ("temperature", not "ed_temperature" or
    # "triage_temperature"), so calling these helpers directly would
    # silently no-op (column_type resolves to None, nothing matches) rather
    # than error. Prefix, clean, then strip the prefix back off before
    # handing to merge_vitalsign_sequence_on_edstay, which expects the bare
    # names.
    vs_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
    vitalsign_raw = vitalsign_raw.rename(columns={c: f"vs_{c}" for c in vs_cols})
    vitalsign_raw = convert_temp_to_celcius(vitalsign_raw)
    vitalsign_raw = remove_outliers(vitalsign_raw, VITALS_VALID_RANGE)
    vitalsign_raw = vitalsign_raw.rename(columns={f"vs_{c}": c for c in vs_cols})
    seq_by_stay = merge_vitalsign_sequence_on_edstay(vitalsign_raw, t_max=T_MAX_ED)

    for split_name, split_df in [("train_v2", train_period), ("test_v2", test_period)]:
        print(f"\nBuilding irregular sequences for {split_name} ({len(split_df)} rows)...")
        values, obs_mask, pad_mask, delta_hours = build_irregular_sequences(
            split_df.reset_index(drop=True), triage_raw, seq_by_stay)
        np.savez_compressed(
            os.path.join(DATA_DIR, f"m2_irregular_sequences_{split_name}.npz"),
            values=values, obs_mask=obs_mask, pad_mask=pad_mask, delta_hours=delta_hours,
            stay_id=split_df["stay_id"].values,
        )
        print(f"Wrote data/m2_irregular_sequences_{split_name}.npz")

    print("\nPhase 2 cohort rebuild complete.")


if __name__ == "__main__":
    main()
