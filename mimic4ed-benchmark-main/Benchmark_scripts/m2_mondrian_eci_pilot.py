"""
M2 -- Phase 0 pilot for Mondrian-ECI sets (GO/NO-GO)
==============================================================================
Runs the new deployable online label-conditional construction
(m2_mondrian_conformal.mondrian_eci_sets) on the v2 chronological-primary
split and evaluates it against the PRE-REGISTERED criteria fixed in the
session plan BEFORE any result was seen.

Two correctness tests run first and must both pass:
  T1 CAUSALITY      -- the radii in force at position t are unchanged when
                       every label at position >= t is flipped, i.e. no
                       future label can influence a current prediction set.
  T2 STATIC EQUIV   -- substituting the static per-class calibration
                       quantile for the online radius reproduces
                       mondrian_lac_sets() bit-for-bit, confirming the two
                       constructions differ in exactly one variable.

Pre-registered endpoints (plan Phase 0):
  PRIMARY   P(C == {1} | y = 1)     -- informative minority coverage
  also      class-1 coverage P(1 in C | y=1), mean |C| overall/by label,
            rates of {1}/{0}/{0,1}/empty, q1 saturation fraction and max.
  Empty sets count as a MISS for both classes.

GO/NO-GO (primary model = Hybrid GRU-D v2):
  NO-GO  mean|C| >= 1.90  OR  q1>=1 on >25% of stream  OR  P(C={1}|y=1) < 0.10
  GO     class-1 coverage in [0.85, 0.95] AND mean|C| < 1.52 (Mondrian-LAC
         anchor) AND P(C={1}|y=1) >= Mondrian-LAC's + 0.05
  MIDDLE otherwise -> contribution reframes to shift-tracking.

Learning rate: swept over the same grid run_eci_4way() uses. The reported
GO/NO-GO uses the lr selected on the calibration_selection fold (train
period, never the test window); the full grid is printed as a sensitivity
check, because saturation is directly lr-driven and a single unlucky lr
must not be allowed to kill the method.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_CSV = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_pilot.csv")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_mondrian_eci_pilot_verdict.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, DEVICE
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2,
    prepare_static_features_v2,
)
from m2_mondrian_conformal import mondrian_lac_sets, mondrian_eci_sets
from m2_conformal_v2 import get_predictions_v2, SEED_DIRS_V2, FINAL_SEEDS

PRIMARY_MODEL = "Hybrid GRU-D v2"
LR_GRID = [0.01, 0.05, 0.1]
ALPHA = 0.1

# Pre-registered thresholds -- do not edit after first run.
NOGO_MEAN_SETSIZE = 1.90
NOGO_SATURATION = 0.25
NOGO_INFORMATIVE = 0.10
GO_COV_LO, GO_COV_HI = 0.85, 0.95
GO_INFORMATIVE_MARGIN = 0.05


def set_metrics(sets, y_true, q=None):
    """Pre-registered metric block. `sets` is (n,2) boolean."""
    y = np.asarray(y_true, dtype=int)
    inc0, inc1 = sets[:, 0], sets[:, 1]
    size = sets.sum(axis=1)
    is_singleton_1 = inc1 & ~inc0
    is_singleton_0 = inc0 & ~inc1
    is_both = inc0 & inc1
    is_empty = ~inc0 & ~inc1

    pos = y == 1
    neg = y == 0
    # Empty set counts as a miss: covered iff the true label is in the set.
    covered = sets[np.arange(len(y)), y]

    m = {
        "coverage_marginal": float(covered.mean()),
        "coverage_class1": float(covered[pos].mean()),
        "coverage_class0": float(covered[neg].mean()),
        "informative_class1": float(is_singleton_1[pos].mean()),  # PRIMARY endpoint
        "mean_set_size": float(size.mean()),
        "mean_set_size_class1": float(size[pos].mean()),
        "mean_set_size_class0": float(size[neg].mean()),
        "rate_singleton_1": float(is_singleton_1.mean()),
        "rate_singleton_0": float(is_singleton_0.mean()),
        "rate_both": float(is_both.mean()),
        "rate_empty": float(is_empty.mean()),
    }
    if q is not None:
        m["q1_saturated_frac"] = float((q[:, 1] >= 1.0).mean())
        m["q1_max"] = float(q[:, 1].max())
        m["q0_saturated_frac"] = float((q[:, 0] >= 1.0).mean())
    return m


def test_causality(p_stream, y_stream, n_cal, lr):
    """T1: flipping every label from position t onward must not change the
    radii in force at positions <= t."""
    n = len(p_stream)
    t = int(n * 0.6)
    _, q_a, _ = mondrian_eci_sets(p_stream, y_stream, n_cal=0, lr=lr, alpha=ALPHA)
    y_b = y_stream.copy()
    y_b[t:] = 1 - y_b[t:]
    _, q_b, _ = mondrian_eci_sets(p_stream, y_b, n_cal=0, lr=lr, alpha=ALPHA)
    same = np.allclose(q_a[: t + 1], q_b[: t + 1], rtol=0, atol=0)
    differs_after = not np.allclose(q_a[t + 1 :], q_b[t + 1 :], rtol=0, atol=0)
    return same, differs_after


def test_static_equivalence(p_final_cal, y_final_cal, p_test):
    """T2: applying the construction's own set rule to the STATIC per-class
    quantile must reproduce mondrian_lac_sets() exactly."""
    ref_sets, q_static = mondrian_lac_sets(p_final_cal, y_final_cal, p_test)
    s_0, s_1 = p_test, 1.0 - p_test
    rebuilt = np.column_stack([s_0 <= q_static[0], s_1 <= q_static[1]])
    return bool(np.array_equal(ref_sets, rebuilt))


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
    print(f"calibration_selection={len(df_cal_sel)}, final_conformal_calibration={len(df_final_cal)}, "
          f"test={len(df_test_sorted)}")

    Xc, Mc, Dc, Pc, yc = prepare_temporal_tensors_v2(df_cal_sel, "train_v2", device=DEVICE)
    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(df_final_cal, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_test_sorted, "test_v2", device=DEVICE)
    Sc = torch.tensor(prepare_static_features_v2(df_cal_sel), device=DEVICE)
    Sf = torch.tensor(prepare_static_features_v2(df_final_cal), device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_test_sorted), device=DEVICE)
    yc_np = yc.cpu().numpy().astype(int)
    yf_np = yf.cpu().numpy().astype(int)
    yt_np = yt.cpu().numpy().astype(int)

    rows = []
    tests_done = False
    for model_name in SEED_DIRS_V2:
        print(f"\n{'='*64}\n{model_name}\n{'='*64}")
        for seed in FINAL_SEEDS:
            p_cal_sel = get_predictions_v2(model_name, seed, Xc, Mc, Dc, Pc, S=Sc)
            p_final_cal = get_predictions_v2(model_name, seed, Xf, Mf, Df, Pf, S=Sf)
            p_test = get_predictions_v2(model_name, seed, Xt, Mt, Dt, Pt, S=St)

            # Static anchor (construction 1), same calibration fold.
            lac_sets, _ = mondrian_lac_sets(p_final_cal, yf_np, p_test)
            lac_m = set_metrics(lac_sets, yt_np)

            if not tests_done:
                print("\n--- correctness tests ---")
                ok2 = test_static_equivalence(p_final_cal, yf_np, p_test)
                print(f"T2 static-equivalence reproduces mondrian_lac_sets exactly: {ok2}")
                assert ok2, "T2 FAILED: set rule diverges from construction 1. STOP."
                p_stream_t = np.concatenate([p_final_cal, p_test])
                y_stream_t = np.concatenate([yf_np, yt_np])
                ok1, changed = test_causality(p_stream_t, y_stream_t, len(p_final_cal), lr=0.1)
                print(f"T1 causality  radii at t unaffected by labels >= t: {ok1} "
                      f"(and DO change afterwards: {changed})")
                assert ok1, "T1 FAILED: future labels leak into current radii. STOP."
                tests_done = True
                print("--- both tests passed ---\n")

            # lr selection on calibration_selection only (never the test window):
            # stream = cal_sel (warm-up) + cal_sel evaluated, pick lr whose
            # class-1 coverage is closest to target, tie-broken on set size.
            best_lr, best_key = None, None
            for lr in LR_GRID:
                half = len(p_cal_sel) // 2
                s_sel, q_sel, _ = mondrian_eci_sets(
                    p_cal_sel, yc_np, n_cal=half, lr=lr, alpha=ALPHA)
                m_sel = set_metrics(s_sel, yc_np[half:], q_sel)
                key = (abs(m_sel["coverage_class1"] - (1 - ALPHA)), m_sel["mean_set_size"])
                if best_key is None or key < best_key:
                    best_key, best_lr = key, lr

            # Full lr grid on the reported stream, for the sensitivity table.
            p_stream = np.concatenate([p_final_cal, p_test])
            y_stream = np.concatenate([yf_np, yt_np])
            for lr in LR_GRID:
                eci_sets, q_test, diag = mondrian_eci_sets(
                    p_stream, y_stream, n_cal=len(p_final_cal), lr=lr, alpha=ALPHA)
                m = set_metrics(eci_sets, yt_np, q_test)
                rows.append({
                    "model": model_name, "seed": seed, "lr": lr,
                    "selected_lr": (lr == best_lr),
                    **{f"eci_{k}": v for k, v in m.items()},
                    **{f"lac_{k}": v for k, v in lac_m.items()},
                })
                tag = " <-- selected" if lr == best_lr else ""
                print(f"  seed {seed} lr={lr:<5} cov1={m['coverage_class1']:.4f} "
                      f"informative1={m['informative_class1']:.4f} |C|={m['mean_set_size']:.4f} "
                      f"empty={m['rate_empty']:.4f} q1sat={m['q1_saturated_frac']:.3f}{tag}", flush=True)
            print(f"           Mondrian-LAC anchor: cov1={lac_m['coverage_class1']:.4f} "
                  f"informative1={lac_m['informative_class1']:.4f} |C|={lac_m['mean_set_size']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    # ---- GO/NO-GO on the primary model at its selected lr ----------------
    sel = df[(df["model"] == PRIMARY_MODEL) & (df["selected_lr"])]
    assert len(sel) == len(FINAL_SEEDS), f"expected {len(FINAL_SEEDS)} selected rows, got {len(sel)}"
    mean_size = sel["eci_mean_set_size"].mean()
    sat = sel["eci_q1_saturated_frac"].mean()
    informative = sel["eci_informative_class1"].mean()
    cov1 = sel["eci_coverage_class1"].mean()
    lac_informative = sel["lac_informative_class1"].mean()
    lac_size = sel["lac_mean_set_size"].mean()

    nogo = (mean_size >= NOGO_MEAN_SETSIZE) or (sat > NOGO_SATURATION) or (informative < NOGO_INFORMATIVE)
    go = ((GO_COV_LO <= cov1 <= GO_COV_HI)
          and (mean_size < lac_size)
          and (informative >= lac_informative + GO_INFORMATIVE_MARGIN))
    verdict = "NO-GO" if nogo else ("GO" if go else "MIDDLE")

    print(f"\n{'='*64}\nPRE-REGISTERED VERDICT ({PRIMARY_MODEL}, 5 seeds, selected lr)\n{'='*64}")
    print(f"  class-1 coverage          {cov1:.4f}   (GO band [{GO_COV_LO}, {GO_COV_HI}])")
    print(f"  informative P(C={{1}}|y=1)  {informative:.4f}   (NO-GO if < {NOGO_INFORMATIVE}; "
          f"Mondrian-LAC {lac_informative:.4f}, need >= {lac_informative + GO_INFORMATIVE_MARGIN:.4f})")
    print(f"  mean |C|                  {mean_size:.4f}   (NO-GO if >= {NOGO_MEAN_SETSIZE}; "
          f"Mondrian-LAC anchor {lac_size:.4f})")
    print(f"  q1 saturation fraction    {sat:.4f}   (NO-GO if > {NOGO_SATURATION})")
    print(f"\n  VERDICT: {verdict}")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "model": PRIMARY_MODEL, "n_seeds": len(FINAL_SEEDS), "verdict": verdict,
            "coverage_class1": cov1, "informative_class1": informative,
            "mean_set_size": mean_size, "q1_saturated_frac": sat,
            "lac_informative_class1": lac_informative, "lac_mean_set_size": lac_size,
            "thresholds": {
                "nogo_mean_set_size": NOGO_MEAN_SETSIZE, "nogo_saturation": NOGO_SATURATION,
                "nogo_informative": NOGO_INFORMATIVE, "go_cov_band": [GO_COV_LO, GO_COV_HI],
                "go_informative_margin": GO_INFORMATIVE_MARGIN,
            },
        }, f, indent=2)
    print(f"  Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
