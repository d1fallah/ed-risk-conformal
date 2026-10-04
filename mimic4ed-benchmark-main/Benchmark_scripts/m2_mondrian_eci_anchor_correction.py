"""
M2 -- Re-evaluation of the Mondrian-ECI verdict against the CORRECTED anchor
==============================================================================
Written 2026-09-20 after a defect was found in mondrian_lac_sets(), the
static Mondrian-LAC construction used as the comparator ("anchor") in both
the pilot and the confirmatory run (see the docstring of
test_mondrian_lac_validity.py for the defect and its signature).

What is and is not changed. The pre-registered CRITERIA are unchanged and are
imported from the pilot, not re-typed. The Mondrian-ECI sets themselves are
unchanged (the defect was in the anchor, not in mondrian_eci_sets; the
confirmatory ECI rows are reused verbatim, since ECI is deterministic given
the predictions). Only the anchor is recomputed with the fixed function.

Both verdicts are reported side by side, and the original confirmatory verdict
(against the defective anchor) is NOT overwritten: it is retained in
m2_v2_mondrian_eci_confirmatory_verdict.json with a note added to the
pre-registration document. Reporting only the corrected verdict would hide
that the original comparison was made against a comparator that could never
emit a confident positive.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
IN_CSV = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_confirmatory.csv")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_anchor_corrected.csv")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_anchor_corrected_verdict.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
)
from m2_mondrian_conformal import mondrian_lac_sets
from m2_conformal_v2 import get_predictions_v2, SEED_DIRS_V2, FINAL_SEEDS
from m2_mondrian_eci_pilot import (
    set_metrics, PRIMARY_MODEL, NOGO_MEAN_SETSIZE, NOGO_SATURATION, NOGO_INFORMATIVE,
    GO_COV_LO, GO_COV_HI, GO_INFORMATIVE_MARGIN,
)

TEST_PREVALENCE = None  # filled from data


def verdict(cov1, size, sat, informative, a_size, a_inf):
    nogo = (size >= NOGO_MEAN_SETSIZE) or (sat > NOGO_SATURATION) or (informative < NOGO_INFORMATIVE)
    go = (GO_COV_LO <= cov1 <= GO_COV_HI) and (size < a_size) and (informative >= a_inf + GO_INFORMATIVE_MARGIN)
    return "NO-GO" if nogo else ("GO" if go else "MIDDLE")


def main():
    conf = pd.read_csv(IN_CSV)
    conf = conf[conf["selected_lr"]].copy()

    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev = load_dev_pool_split_v2()
    final_ids = set(dev["final_conformal_calibration"]["subject_ids"].tolist())
    df_final = sort_chronologically(df_train[df_train["subject_id"].isin(final_ids)])
    df_test_s = sort_chronologically(df_test)

    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_final, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_test_s, "test_v2", device=DEVICE)
    Sf = torch.tensor(prepare_static_features_v2(df_final), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_test_s), device=DEVICE)
    yf_np, yt_np = yf.cpu().numpy().astype(int), yt.cpu().numpy().astype(int)
    prev = float(yt_np.mean())

    rows = []
    for model_name in SEED_DIRS_V2:
        for seed in FINAL_SEEDS:
            p_final = get_predictions_v2(model_name, seed, Xf, Mf, Df, Pf, S=Sf)
            p_test = get_predictions_v2(model_name, seed, Xt, Mt, Dt, Pt, S=St)
            lac_sets, q = mondrian_lac_sets(p_final, yf_np, p_test)
            m = set_metrics(lac_sets, yt_np)
            r = conf[(conf["model"] == model_name) & (conf["seed"] == seed)].iloc[0]
            row = {"model": model_name, "seed": seed, "lr": r["lr"],
                   **{f"eci_{k}": r[f"eci_{k}"] for k in
                      ("coverage_class0", "coverage_class1", "informative_class1", "mean_set_size",
                       "rate_singleton_1", "rate_both", "rate_empty", "q1_saturated_frac")},
                   **{f"lacfix_{k}": v for k, v in m.items()},
                   "lacold_informative_class1": r["lac_informative_class1"],
                   "lacold_mean_set_size": r["lac_mean_set_size"],
                   "lacold_coverage_class0": r["lac_coverage_class0"],
                   "lacold_coverage_class1": r["lac_coverage_class1"]}
            # precision of a confident positive: P(y=1 | C={1}) = P(C={1}|y=1)*prev / P(C={1})
            for tag, inf, s1 in (("eci", row["eci_informative_class1"], row["eci_rate_singleton_1"]),
                                 ("lacfix", m["informative_class1"], m["rate_singleton_1"])):
                row[f"{tag}_ppv_singleton1"] = (inf * prev / s1) if s1 > 0 else float("nan")
            rows.append(row)
        print(f"  {model_name}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nTest-window class-1 prevalence: {prev:.4f}\n")
    cols = ["coverage_class0", "coverage_class1", "informative_class1", "mean_set_size", "rate_singleton_1", "rate_both"]
    for mn in SEED_DIRS_V2:
        d = df[df.model == mn]
        print(f"--- {mn} (5-seed means) ---")
        print(f"{'':28s}{'DEFECTIVE anchor':>18s}{'CORRECTED anchor':>18s}{'Mondrian-ECI':>16s}")
        for c in cols:
            old = d[f"lacold_{c}"].mean() if f"lacold_{c}" in d else float("nan")
            print(f"{c:28s}{old:>18.4f}{d[f'lacfix_{c}'].mean():>18.4f}{d[f'eci_{c}'].mean():>16.4f}")
        print(f"{'PPV of a confident {1}':28s}{'n/a':>18s}{d['lacfix_ppv_singleton1'].mean():>18.4f}{d['eci_ppv_singleton1'].mean():>16.4f}")
        print()

    d = df[df.model == PRIMARY_MODEL]
    cov1 = d["eci_coverage_class1"].mean(); size = d["eci_mean_set_size"].mean()
    sat = d["eci_q1_saturated_frac"].mean(); inf_ = d["eci_informative_class1"].mean()
    v_old = verdict(cov1, size, sat, inf_, d["lacold_mean_set_size"].mean(), d["lacold_informative_class1"].mean())
    v_new = verdict(cov1, size, sat, inf_, d["lacfix_mean_set_size"].mean(), d["lacfix_informative_class1"].mean())

    print("=" * 70)
    print(f"PRE-REGISTERED CRITERIA, unchanged, {PRIMARY_MODEL}, 5 seeds, guarded lr")
    print("=" * 70)
    print(f"  Mondrian-ECI:  class-1 cov {cov1:.4f} | informative {inf_:.4f} | |C| {size:.4f} | q1 sat {sat:.4f}")
    print(f"  vs DEFECTIVE anchor  (|C| {d['lacold_mean_set_size'].mean():.4f}, informative {d['lacold_informative_class1'].mean():.4f}) -> {v_old}")
    print(f"  vs CORRECTED anchor  (|C| {d['lacfix_mean_set_size'].mean():.4f}, informative {d['lacfix_informative_class1'].mean():.4f}) -> {v_new}")
    print(f"    GO needs |C| < {d['lacfix_mean_set_size'].mean():.4f} and informative >= "
          f"{d['lacfix_informative_class1'].mean() + GO_INFORMATIVE_MARGIN:.4f}")

    with open(OUT_JSON, "w") as f:
        json.dump({"model": PRIMARY_MODEL, "verdict_vs_defective_anchor": v_old,
                   "verdict_vs_corrected_anchor": v_new, "test_prevalence": prev,
                   "eci": {"coverage_class1": cov1, "informative_class1": inf_, "mean_set_size": size, "q1_saturated_frac": sat},
                   "anchor_defective": {"mean_set_size": d["lacold_mean_set_size"].mean(),
                                        "informative_class1": d["lacold_informative_class1"].mean()},
                   "anchor_corrected": {"mean_set_size": d["lacfix_mean_set_size"].mean(),
                                        "informative_class1": d["lacfix_informative_class1"].mean(),
                                        "coverage_class0": d["lacfix_coverage_class0"].mean(),
                                        "coverage_class1": d["lacfix_coverage_class1"].mean()}}, f, indent=2)
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
