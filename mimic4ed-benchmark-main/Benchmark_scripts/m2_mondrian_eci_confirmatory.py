"""
M2 -- Mondrian-ECI sets: CONFIRMATORY run (guarded learning-rate selection)
==============================================================================
Executes Part 2 of M2_MondrianECI_PreRegistration.md, which was written and
committed to disk BEFORE this script was first run. Read that document first:
it records that the pilot (m2_mondrian_eci_pilot.py) returned NO-GO, why,
and exactly what is -- and is not -- changed here.

The ONE change relative to the pilot: learning-rate selection gains a
degeneracy guard. For each candidate lr on the pre-declared grid, the
construction is run on the calibration_selection fold ONLY (train period;
no test-window data is read during selection), and the candidate is
REJECTED if its q1 saturation fraction on that fold exceeds SAT_GUARD.
Survivors are ranked by the unchanged pilot rule; if every candidate is
rejected the smallest lr is used and the fall-through is recorded.

Everything else is imported from the pilot module rather than re-typed, so
the endpoints, thresholds, grid, seeds and anchor cannot drift between the
two runs. The pilot's own artifacts are not touched.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_confirmatory.csv")
OUT_GUARD_CSV = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_confirmatory_guard_trace.csv")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_confirmatory_verdict.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2,
    prepare_static_features_v2,
)
from m2_mondrian_conformal import mondrian_lac_sets, mondrian_eci_sets
from m2_conformal_v2 import get_predictions_v2, SEED_DIRS_V2, FINAL_SEEDS
# Single source of truth for everything the pre-registration says is unchanged.
from m2_mondrian_eci_pilot import (
    set_metrics, test_causality, test_static_equivalence, PRIMARY_MODEL, LR_GRID, ALPHA,
    NOGO_MEAN_SETSIZE, NOGO_SATURATION, NOGO_INFORMATIVE,
    GO_COV_LO, GO_COV_HI, GO_INFORMATIVE_MARGIN,
)

# The guard threshold is numerically the SAME as the declared NO-GO saturation
# criterion, so the guard and the verdict cannot disagree about "degenerate".
SAT_GUARD = NOGO_SATURATION


def select_lr_guarded(p_cal_sel, y_cal_sel):
    """Guarded lr selection on the calibration_selection fold ONLY.
    Returns (chosen_lr, trace_rows, fell_through)."""
    half = len(p_cal_sel) // 2
    trace = []
    for lr in LR_GRID:
        s_sel, q_sel, _ = mondrian_eci_sets(p_cal_sel, y_cal_sel, n_cal=half, lr=lr, alpha=ALPHA)
        m = set_metrics(s_sel, y_cal_sel[half:], q_sel)
        trace.append({
            "lr": lr,
            "sel_coverage_class1": m["coverage_class1"],
            "sel_mean_set_size": m["mean_set_size"],
            "sel_q1_saturated_frac": m["q1_saturated_frac"],
            "rejected_by_guard": m["q1_saturated_frac"] > SAT_GUARD,
        })
    survivors = [t for t in trace if not t["rejected_by_guard"]]
    if not survivors:
        return min(LR_GRID), trace, True
    best = min(survivors, key=lambda t: (abs(t["sel_coverage_class1"] - (1 - ALPHA)), t["sel_mean_set_size"]))
    return best["lr"], trace, False


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

    Xc, Mc, Dc, Pc, yc = prepare_temporal_tensors_v2(df_cal_sel, "train_v2", device=DEVICE)
    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_final_cal, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_test_sorted, "test_v2", device=DEVICE)
    Sc = torch.tensor(prepare_static_features_v2(df_cal_sel), device=DEVICE)
    Sf = torch.tensor(prepare_static_features_v2(df_final_cal), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_test_sorted), device=DEVICE)
    yc_np, yf_np, yt_np = (a.cpu().numpy().astype(int) for a in (yc, yf, yt))

    rows, guard_rows = [], []
    tests_done = False
    for model_name in SEED_DIRS_V2:
        print(f"\n{'='*64}\n{model_name}\n{'='*64}")
        for seed in FINAL_SEEDS:
            p_cal_sel = get_predictions_v2(model_name, seed, Xc, Mc, Dc, Pc, S=Sc)
            p_final_cal = get_predictions_v2(model_name, seed, Xf, Mf, Df, Pf, S=Sf)
            p_test = get_predictions_v2(model_name, seed, Xt, Mt, Dt, Pt, S=St)

            lac_sets, _ = mondrian_lac_sets(p_final_cal, yf_np, p_test)
            lac_m = set_metrics(lac_sets, yt_np)

            if not tests_done:
                assert test_static_equivalence(p_final_cal, yf_np, p_test), "T2 FAILED"
                ok1, changed = test_causality(np.concatenate([p_final_cal, p_test]),
                                              np.concatenate([yf_np, yt_np]), len(p_final_cal), lr=0.1)
                assert ok1 and changed, "T1 FAILED"
                print("correctness tests T1 + T2 passed")
                tests_done = True

            chosen_lr, trace, fell_through = select_lr_guarded(p_cal_sel, yc_np)
            for t in trace:
                guard_rows.append({"model": model_name, "seed": seed, **t,
                                   "chosen": t["lr"] == chosen_lr, "all_rejected_fallthrough": fell_through})

            p_stream = np.concatenate([p_final_cal, p_test])
            y_stream = np.concatenate([yf_np, yt_np])
            for lr in LR_GRID:
                eci_sets, q_test, _ = mondrian_eci_sets(p_stream, y_stream, n_cal=len(p_final_cal),
                                                         lr=lr, alpha=ALPHA)
                m = set_metrics(eci_sets, yt_np, q_test)
                rows.append({"model": model_name, "seed": seed, "lr": lr, "selected_lr": lr == chosen_lr,
                             "fell_through": fell_through,
                             **{f"eci_{k}": v for k, v in m.items()},
                             **{f"lac_{k}": v for k, v in lac_m.items()}})
            rej = [t["lr"] for t in trace if t["rejected_by_guard"]]
            print(f"  seed {seed}: guard rejected lr={rej or 'none'} -> chose lr={chosen_lr}"
                  f"{'  [ALL REJECTED: fell through]' if fell_through else ''}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(guard_rows).to_csv(OUT_GUARD_CSV, index=False)
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_GUARD_CSV}")

    sel = df[(df["model"] == PRIMARY_MODEL) & (df["selected_lr"])]
    assert len(sel) == len(FINAL_SEEDS)
    mean_size = sel["eci_mean_set_size"].mean()
    sat = sel["eci_q1_saturated_frac"].mean()
    informative = sel["eci_informative_class1"].mean()
    cov1 = sel["eci_coverage_class1"].mean()
    lac_informative = sel["lac_informative_class1"].mean()
    lac_size = sel["lac_mean_set_size"].mean()

    nogo = (mean_size >= NOGO_MEAN_SETSIZE) or (sat > NOGO_SATURATION) or (informative < NOGO_INFORMATIVE)
    go = ((GO_COV_LO <= cov1 <= GO_COV_HI) and (mean_size < lac_size)
          and (informative >= lac_informative + GO_INFORMATIVE_MARGIN))
    verdict = "NO-GO" if nogo else ("GO" if go else "MIDDLE")
    chosen = sel.sort_values("seed")["lr"].tolist()

    print(f"\n{'='*64}\nCONFIRMATORY VERDICT ({PRIMARY_MODEL}, 5 seeds, guarded lr)\n{'='*64}")
    print(f"  lr chosen per seed        {chosen}")
    print(f"  class-1 coverage          {cov1:.4f}   (GO band [{GO_COV_LO}, {GO_COV_HI}])")
    print(f"  informative P(C={{1}}|y=1)  {informative:.4f}   (anchor {lac_informative:.4f}, need >= {lac_informative + GO_INFORMATIVE_MARGIN:.4f})")
    print(f"  mean |C|                  {mean_size:.4f}   (anchor {lac_size:.4f}; NO-GO if >= {NOGO_MEAN_SETSIZE})")
    print(f"  q1 saturation fraction    {sat:.4f}   (NO-GO if > {NOGO_SATURATION})")
    print(f"\n  VERDICT: {verdict}")

    # Secondary models, reported but not gating.
    print("\nSecondary models (informational, not part of the pre-registered verdict):")
    for mn in SEED_DIRS_V2:
        if mn == PRIMARY_MODEL:
            continue
        s = df[(df["model"] == mn) & (df["selected_lr"])]
        print(f"  {mn}: cov1={s['eci_coverage_class1'].mean():.4f} "
              f"informative={s['eci_informative_class1'].mean():.4f} "
              f"|C|={s['eci_mean_set_size'].mean():.4f} (anchor {s['lac_mean_set_size'].mean():.4f}) "
              f"q1sat={s['eci_q1_saturated_frac'].mean():.4f}")

    with open(OUT_JSON, "w") as f:
        json.dump({"model": PRIMARY_MODEL, "verdict": verdict, "lr_chosen_per_seed": chosen,
                   "coverage_class1": cov1, "informative_class1": informative,
                   "mean_set_size": mean_size, "q1_saturated_frac": sat,
                   "lac_informative_class1": lac_informative, "lac_mean_set_size": lac_size,
                   "sat_guard": SAT_GUARD}, f, indent=2)
    print(f"  Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
