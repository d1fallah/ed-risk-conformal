"""
M2 Hybrid Guide v3.8 -- Conformal Layer (Sec 2.4/8)
========================================================================
Applies both conformal procedures to the 3 GRU-D-family models --
Temporal-Only GRU-D, Frozen-Temporal Head Control, and Hybrid GRU-D --
using the fixed 4-way development-pool split's `calibration_selection` and
`final_conformal_calibration` subsets (Sec 5.1/2.4). Static-Only MLP and
XGBoost-Matched are untouched here (no conformal reporting, per Sec 8).

MAPIE (classification prediction sets): m2_hybrid_common.py's
run_mapie_classification(), calibrated on `final_conformal_calibration`
only.

ECI (probability intervals): m2_hybrid_common.py's run_eci_4way() --
lr swept on `calibration_selection`, threshold estimated on
`final_conformal_calibration`, both frozen before test evaluation. This
replaces the pre-v3.8 single-fold ECI approach for these 3 models.

Merges the new coverage/set-size/width columns into each (model,seed)'s
EXISTING row in data/results_hybrid.csv (loaded first, not overwritten --
save_seed_metrics() replaces by (model,seed) key, so the discrimination/
calibration metrics already saved by Phases I/J/K must be re-included here
or they would be silently lost).
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    TARGET, HYBRID_STATIC_COLS, add_hybrid_static_features,
    prepare_temporal_tensors, build_fixed_dev_split,
    load_temporal_only_model, load_frozen_encoder, load_fusion_head,
    predict_proba_temporal_only, predict_proba_frozen,
    run_mapie_classification, run_eci_4way, save_seed_metrics,
    prepare_static_features, conformal_coverage_report, sort_chronologically,
)

FINAL_SEEDS = [42, 123, 7, 456, 789]
SEED_DIRS = {
    "Temporal-Only GRU-D": "temporal_only_grud",
    "Frozen-Temporal Head Control": "frozen_temporal_control",
    "Hybrid GRU-D": "hybrid_grud",
}


def get_predictions(model_name, seed, X, M, D, S=None):
    if model_name == "Temporal-Only GRU-D":
        model = load_temporal_only_model(seed)
        return predict_proba_temporal_only(model, X, M, D)
    encoder = load_frozen_encoder(seed)
    seed_dir = os.path.join(DATA_DIR, "m2_hybrid_seed_results", SEED_DIRS[model_name])
    with open(os.path.join(seed_dir, f"seed_{seed}.json")) as f:
        cfg = json.load(f)["config"]
    if model_name == "Frozen-Temporal Head Control":
        head = load_fusion_head("m2_frozen_temporal_control_head", seed,
                                 static_hidden=None, fusion_dropout=cfg["fusion_dropout"])
        return predict_proba_frozen(encoder, head, X, M, D)
    else:  # Hybrid GRU-D
        head = load_fusion_head("m2_hybrid_grud_head", seed, static_hidden=cfg["static_hidden"],
                                 static_dropout=cfg["static_dropout"], fusion_dropout=cfg["fusion_dropout"])
        return predict_proba_frozen(encoder, head, X, M, D, S=S)


def main():
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_train = add_hybrid_static_features(df_train)
    df_test = add_hybrid_static_features(df_test)

    dev_split = build_fixed_dev_split()
    cal_sel_ids = set(dev_split["calibration_selection"]["subject_ids"].tolist())
    final_cal_ids = set(dev_split["final_conformal_calibration"]["subject_ids"].tolist())
    # ECI is an ONLINE method for sequentially-arriving, non-exchangeable
    # data -- its entire justification in this module is adaptation under
    # temporal non-stationarity. Run over rows in file order it degenerates
    # into an expensive split-conformal, which is precisely what the
    # pre-Deviation-7 pipeline did: test.csv is NOT sorted by intime
    # (verified directly), and neither subset was ordered before being
    # concatenated into ECI's stream. Every stream is now in true arrival
    # order, matching what m2_ablations.py's Track A path already did.
    #
    # LIMITATION, deliberately not papered over: the outer train/test split
    # is a RANDOM patient-level split (data_general_processing.py), so the
    # calibration prefix and the test suffix are temporally interleaved
    # populations rather than a past->future sequence. Ordering within each
    # stream is necessary but not sufficient to exhibit the distribution
    # shift ECI exists to handle; a genuinely chronological split is the
    # separate, larger change tracked as an open item.
    # CORRECTION: an earlier version of this fix sorted by `intime`. That is
    # wrong for MIMIC-IV -- dates are shifted by a random per-patient offset
    # (anchor_year spans 2110-2208 here), so `intime` is not comparable
    # across patients and sorting by it yields an arbitrary order that only
    # looks chronological. sort_chronologically() uses the de-shifted
    # approximate real year instead; see _load_master_vitals_lookup's
    # sibling _load_real_time_lookup() for the recipe and its +/-1 year
    # resolution limit.
    df_cal_sel = sort_chronologically(df_train[df_train["subject_id"].isin(cal_sel_ids)])
    df_final_cal = sort_chronologically(df_train[df_train["subject_id"].isin(final_cal_ids)])
    df_test = sort_chronologically(df_test)
    print(f"calibration_selection: {len(df_cal_sel)} rows, "
          f"final_conformal_calibration: {len(df_final_cal)} rows, test: {len(df_test)} rows")

    Xc, Mc, Dc, yc = prepare_temporal_tensors(df_cal_sel)
    Xf, Mf, Df, yf = prepare_temporal_tensors(df_final_cal)
    Xt, Mt, Dt, yt = prepare_temporal_tensors(df_test)
    # Must use the SAME shared scaler the Hybrid was trained with.
    Sc = prepare_static_features(df_cal_sel)
    Sf = prepare_static_features(df_final_cal)
    St = prepare_static_features(df_test)

    all_results = []
    for model_name in SEED_DIRS:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        for seed in FINAL_SEEDS:
            p_cal_sel = get_predictions(model_name, seed, Xc, Mc, Dc, S=Sc)
            p_final_cal = get_predictions(model_name, seed, Xf, Mf, Df, S=Sf)
            p_test = get_predictions(model_name, seed, Xt, Mt, Dt, S=St)

            # MAPIE: classification prediction sets, calibrated on final_conformal_calibration
            sets = run_mapie_classification(p_final_cal, yf, p_test)
            set_sizes = sets.sum(axis=1)
            mapie_cov = conformal_coverage_report(
                yt, sets[np.arange(len(yt)), yt], set_size=set_sizes, prefix="mapie_")
            coverage_mapie = mapie_cov["mapie_coverage"]
            set_size_mapie = float(set_sizes.mean())

            # ECI: lr selected on calibration_selection, threshold on final_conformal_calibration
            eci_lr, coverage_eci, width_eci, lo_t, hi_t, eci_candidates = run_eci_4way(
                p_cal_sel, yc, p_final_cal, yf, p_test, yt,
            )
            eci_cov = conformal_coverage_report(
                yt, (yt >= lo_t) & (yt <= hi_t), prefix="eci_")

            print(f"  seed={seed}: MAPIE cov={coverage_mapie:.4f} "
                  f"(class0={mapie_cov['mapie_coverage_class0']:.4f} "
                  f"class1={mapie_cov['mapie_coverage_class1']:.4f}, "
                  f"empty={mapie_cov['mapie_empty_set_rate']:.4f}) set_size={set_size_mapie:.4f}")
            print(f"           ECI(lr={eci_lr}) cov={coverage_eci:.4f} "
                  f"(class0={eci_cov['eci_coverage_class0']:.4f} "
                  f"class1={eci_cov['eci_coverage_class1']:.4f}) width={width_eci:.4f}")

            conformal_metrics = {
                "coverage_mapie": coverage_mapie, "set_size_mapie": set_size_mapie,
                "eci_lr_selected": eci_lr, "coverage_eci": coverage_eci, "width_eci": width_eci,
                **{k: v for k, v in mapie_cov.items() if k != "mapie_coverage"},
                **{k: v for k, v in eci_cov.items() if k != "eci_coverage"},
            }

            # Merge with the discrimination/calibration metrics already saved
            # by Phases I/J/K -- save_seed_metrics() replaces by (model,seed),
            # so omitting the existing fields here would silently drop them.
            seed_dir = os.path.join(DATA_DIR, "m2_hybrid_seed_results", SEED_DIRS[model_name])
            seed_json_path = os.path.join(seed_dir, f"seed_{seed}.json")
            with open(seed_json_path) as f:
                seed_result = json.load(f)
            seed_result["metrics"].update(conformal_metrics)
            seed_result["eci_stage1_candidates"] = eci_candidates
            with open(seed_json_path, "w") as f:
                json.dump(seed_result, f, indent=2)

            save_seed_metrics(model_name, seed, seed_result["metrics"])
            all_results.append({"model": model_name, "seed": seed, **conformal_metrics})

    df_conf = pd.DataFrame(all_results)
    print(f"\n{'='*60}\nConformal summary (mean +/- SD across seeds)\n{'='*60}")
    summary = df_conf.groupby("model").agg(
        coverage_mapie_mean=("coverage_mapie", "mean"), coverage_mapie_sd=("coverage_mapie", "std"),
        mapie_cov_class0_mean=("mapie_coverage_class0", "mean"),
        mapie_cov_class1_mean=("mapie_coverage_class1", "mean"),
        mapie_empty_rate_mean=("mapie_empty_set_rate", "mean"),
        set_size_mapie_mean=("set_size_mapie", "mean"), set_size_mapie_sd=("set_size_mapie", "std"),
        coverage_eci_mean=("coverage_eci", "mean"), coverage_eci_sd=("coverage_eci", "std"),
        eci_cov_class0_mean=("eci_coverage_class0", "mean"),
        eci_cov_class1_mean=("eci_coverage_class1", "mean"),
        width_eci_mean=("width_eci", "mean"), width_eci_sd=("width_eci", "std"),
    )
    print(summary.to_string())
    summary.to_csv(os.path.join(DATA_DIR, "m2_hybrid_conformal_summary.csv"))


if __name__ == "__main__":
    main()
