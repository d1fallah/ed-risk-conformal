"""
M2 -- Conformal layer, irregular-series (v2) cohort (Phase 4)
==============================================================================
v2 counterpart of m2_hybrid_conformal.py: applies MAPIE (classification
prediction sets) and two-stage ECI (probability intervals) to the 3
GRU-D-family v2 models (Temporal-Only GRU-D v2, Frozen-Temporal Head
Control v2, Hybrid GRU-D v2), using the v2 dev-pool's
calibration_selection / final_conformal_calibration subject subsets.
Scope matches the T=2 pipeline's own conformal scope (Static-Only MLP and
XGBoost-Matched are not conformalized there either).

Requires all 3 models' 5-seed checkpoints to already exist
(m2_temporal_only_grud_v2.py, m2_frozen_temporal_control_v2.py,
m2_hybrid_grud_v2.py).

Also runs the Mondrian (label-conditional) construction from
m2_mondrian_conformal.py's mondrian_lac_sets(), reused directly rather
than reimplemented, so the v2 result set answers Q1-blocker #2 the same
way the T=2 result set does -- not left as a T=2-only fix.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_conformal_summary.csv")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    HYBRID_STATIC_COLS, add_hybrid_static_features, run_mapie_classification, run_eci_4way,
    conformal_coverage_report, sort_chronologically, DEVICE,
)
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
    load_temporal_only_model_v2, load_frozen_encoder_v2, load_fusion_head_v2,
    predict_proba_temporal_only_v2, predict_proba_frozen_v2,
)
from m2_mondrian_conformal import mondrian_lac_sets

FINAL_SEEDS = [42, 123, 7, 456, 789]
SEED_DIRS_V2 = {
    "Temporal-Only GRU-D v2": "temporal_only_grud",
    "Frozen-Temporal Head Control v2": "frozen_temporal_control",
    "Hybrid GRU-D v2": "hybrid_grud",
}


def _tuned(name):
    """Head architecture actually used to TRAIN the checkpoints, read from the
    Phase-2b tuned configs rather than hardcoded. Dropout rates carry no
    parameters and are inert at eval(), so this does not change any number --
    but hardcoding them here drifted out of sync with the checkpoints once the
    heads were re-tuned (static_dropout 0.1 -> 0.3), which is exactly the kind
    of silent config/artifact divergence this module's deviation log exists to
    prevent. Reading the config makes the divergence impossible."""
    path = os.path.join(DATA_DIR, name)
    assert os.path.exists(path), f"{path} not found -- run the Phase-2b tuning stage first."
    with open(path) as f:
        return json.load(f)


def get_predictions_v2(model_name, seed, X, M, D, P, S=None):
    if model_name == "Temporal-Only GRU-D v2":
        model = load_temporal_only_model_v2(seed)
        return predict_proba_temporal_only_v2(model, X, M, D, P)
    encoder = load_frozen_encoder_v2(seed)
    if model_name == "Frozen-Temporal Head Control v2":
        cfg = _tuned("m2_v2_frozen_control_tuned_config.json")
        head = load_fusion_head_v2("m2_frozen_temporal_control_head", seed, static_hidden=None,
                                    fusion_dropout=cfg["fusion_dropout"])
        return predict_proba_frozen_v2(encoder, head, X, M, D, P)
    else:  # Hybrid GRU-D v2
        cfg = _tuned("m2_v2_hybrid_head_tuned_config.json")
        head = load_fusion_head_v2("m2_hybrid_grud_head", seed, static_hidden=cfg["static_hidden"],
                                    static_dropout=cfg["static_dropout"],
                                    fusion_dropout=cfg["fusion_dropout"])
        return predict_proba_frozen_v2(encoder, head, X, M, D, P, S=S)


def main():
    print("Loading v2 data...")
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))

    dev_split = load_dev_pool_split_v2()
    cal_sel_ids = set(dev_split["calibration_selection"]["subject_ids"].tolist())
    final_cal_ids = set(dev_split["final_conformal_calibration"]["subject_ids"].tolist())

    df_cal_sel = sort_chronologically(df_train[df_train["subject_id"].isin(cal_sel_ids)])
    df_final_cal = sort_chronologically(df_train[df_train["subject_id"].isin(final_cal_ids)])
    df_test_sorted = sort_chronologically(df_test)
    print(f"calibration_selection: {len(df_cal_sel)}, final_conformal_calibration: {len(df_final_cal)}, "
          f"test: {len(df_test_sorted)}")

    Xc, Mc, Dc, Pc, yc = prepare_temporal_tensors_v2(df_cal_sel, "train_v2", device=DEVICE)
    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_final_cal, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_test_sorted, "test_v2", device=DEVICE)
    Sc = torch.tensor(prepare_static_features_v2(df_cal_sel), device=DEVICE)
    Sf = torch.tensor(prepare_static_features_v2(df_final_cal), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_test_sorted), device=DEVICE)

    yc_np, yf_np, yt_np = yc.cpu().numpy().astype(int), yf.cpu().numpy().astype(int), yt.cpu().numpy().astype(int)

    all_results = []
    for model_name in SEED_DIRS_V2:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        for seed in FINAL_SEEDS:
            p_cal_sel = get_predictions_v2(model_name, seed, Xc, Mc, Dc, Pc, S=Sc)
            p_final_cal = get_predictions_v2(model_name, seed, Xf, Mf, Df, Pf, S=Sf)
            p_test = get_predictions_v2(model_name, seed, Xt, Mt, Dt, Pt, S=St)

            # MAPIE (marginal) classification sets
            sets = run_mapie_classification(p_final_cal, yf_np, p_test)
            set_sizes = sets.sum(axis=1)
            mapie_cov = conformal_coverage_report(yt_np, sets[np.arange(len(yt_np)), yt_np],
                                                    set_size=set_sizes, prefix="mapie_")

            # ECI
            eci_lr, coverage_eci, width_eci, lo_t, hi_t, _cands = run_eci_4way(
                p_cal_sel, yc_np, p_final_cal, yf_np, p_test, yt_np,
            )
            eci_cov = conformal_coverage_report(yt_np, (yt_np >= lo_t) & (yt_np <= hi_t), prefix="eci_")

            # Mondrian (label-conditional) MAPIE sets -- reused from Phase 1b's fix,
            # applied here so the v2 result set closes Q1-blocker #2 too.
            mondrian_sets, mondrian_q = mondrian_lac_sets(p_final_cal, yf_np, p_test)
            mondrian_set_sizes = mondrian_sets.sum(axis=1)
            mondrian_cov = conformal_coverage_report(
                yt_np, mondrian_sets[np.arange(len(yt_np)), yt_np],
                set_size=mondrian_set_sizes, prefix="mondrian_lac_")

            row = {"model": model_name, "seed": seed, "eci_lr": eci_lr,
                   "eci_width": width_eci, **mapie_cov, **eci_cov, **mondrian_cov,
                   "mondrian_mean_set_size": float(mondrian_set_sizes.mean())}
            all_results.append(row)
            print(f"  seed {seed}: MAPIE cov={mapie_cov['mapie_coverage']:.4f} "
                  f"class1={mapie_cov['mapie_coverage_class1']:.4f} | "
                  f"ECI cov={eci_cov['eci_coverage']:.4f} class1={eci_cov['eci_coverage_class1']:.4f} | "
                  f"Mondrian-LAC class1={mondrian_cov['mondrian_lac_coverage_class1']:.4f}", flush=True)

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print("\n=== 5-seed summary ===")
    for model_name in SEED_DIRS_V2:
        sub = out_df[out_df.model == model_name]
        print(f"{model_name}: MAPIE cov {sub.mapie_coverage.mean():.4f}+/-{sub.mapie_coverage.std():.4f} "
              f"class1 {sub.mapie_coverage_class1.mean():.4f} | "
              f"ECI cov {sub.eci_coverage.mean():.4f}+/-{sub.eci_coverage.std():.4f} "
              f"class1 {sub.eci_coverage_class1.mean():.4f} | "
              f"Mondrian-LAC class1 {sub.mondrian_lac_coverage_class1.mean():.4f}+/-"
              f"{sub.mondrian_lac_coverage_class1.std():.4f}")


if __name__ == "__main__":
    main()
