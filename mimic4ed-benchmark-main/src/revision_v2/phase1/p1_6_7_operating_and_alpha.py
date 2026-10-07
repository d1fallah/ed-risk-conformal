"""
Phase 1.6 -- clinical operating characteristics and prediction-set efficiency (no cost or utility function is used).
Phase 1.7 -- alpha sweep, alpha in {0.20, 0.10, 0.05, 0.01}; eta (learning rate) re-selected per alpha on calibration_selection only.

Methods: static LAC (marginal), static Mondrian-LAC, ECI sets (marginal LAC score, L = 0), Mondrian-ECI (L = 0).
Models: the three GRU-D-family models, 5 seeds each (stored predictions via the repository's prediction cache).
Per run: marginal / class-0 / class-1 coverage, set-type rates, the 2 x 4 cross-table (set type x true class), PPV and sensitivity of
{1}, false-positive rate, false alarms per 100 patients, abstention by class, and for every class-1 quantile the number of
calibration positives behind it with the feasibility check ceil((n+1)(1-alpha)) <= n.

Checks before any sweep value is trusted (alpha = 0.10 must reproduce the stored results):
  T-static   static LAC coverage / size == rev_e3 `static_lac` rows;
  T-eci      ECI marginal sets == rev_e3 `eci_lac` lag-0 rows (selected lr, coverage, size);
  T-mon      Mondrian-ECI == confirmatory rows at the selected lr.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import rv2_conformal as RC  # noqa: E402
import m2_label_lag as ll  # noqa: E402
from m2_mondrian_eci_pilot import set_metrics  # noqa: E402

ALPHAS = [0.20, 0.10, 0.05, 0.01]
MODELS = ["Temporal-Only GRU-D v2", "Frozen-Temporal Head Control v2", "Hybrid GRU-D v2"]
METHODS = ["static_lac", "static_mondrian_lac", "eci_sets", "mondrian_eci"]


def run_one(P, y, alpha):
    p_cs, p_fc, p_te = P["cs"], P["fc"], P["te"]
    y_cs, y_fc, y_te = y["cs"].astype(int), y["fc"].astype(int), y["te"].astype(int)
    out = {}
    s, q = RC.static_lac_sets(p_fc, y_fc, p_te, alpha); out["static_lac"] = (s, {"q": q, "calib_positives": int(y_fc.sum()), "n_cal": len(y_fc)})
    s, q = RC.mondrian_static_sets(p_fc, y_fc, p_te, alpha)
    out["static_mondrian_lac"] = (s, {"q0": q[0], "q1": q[1], "calib_positives": int(y_fc.sum()), "calib_negatives": int((y_fc == 0).sum())})
    s, q, lr = RC.eci_marginal_sets(p_cs, y_cs, p_fc, y_fc, p_te, y_te, alpha)
    out["eci_sets"] = (s, {"lr": lr, "q_inf_frac": float(np.isinf(q).mean()), "q_max": float(np.max(q))})
    s, q, lr, fell = RC.mondrian_eci_sets_run(p_cs, y_cs, p_fc, y_fc, p_te, y_te, alpha)
    out["mondrian_eci"] = (s, {"lr": lr, "fell_through": fell, "q1_saturated_frac": float((q[:, 1] >= 1.0).mean()),
                               "q0_saturated_frac": float((q[:, 0] >= 1.0).mean()), "q1_max": float(q[:, 1].max()),
                               "selection_class1_positives_in_evaluated_half": int(y_cs[len(y_cs) // 2:].sum()),
                               "stream_class1_positives_calibration_plus_test": int(y_fc.sum() + y_te.sum())})
    return out, y_te


def main():
    preds, y, year = ll.get_all_predictions()
    e3 = pd.read_csv(os.path.join(C.DATA, "m2_v2_rev_e3_adaptive_sets.csv"))
    conf = pd.read_csv(os.path.join(C.DATA, "m2_v2_mondrian_eci_confirmatory.csv"))
    res = {"alphas": ALPHAS, "methods": METHODS, "checks_at_alpha_0.10": [], "runs": []}
    for model in MODELS:
        for seed in C.SEEDS:
            for alpha in ALPHAS:
                out, y_te = run_one(preds[(model, seed)], y, alpha)
                fc_pos = int(y["fc"].sum())
                for meth, (sets, info) in out.items():
                    m = set_metrics(sets, y_te)
                    oc = RC.operating_characteristics(sets, y_te)
                    rec = {"model": model, "seed": seed, "alpha": alpha, "method": meth, "metrics": m, "operating": oc, "info": info,
                           "class1_calibration_positives": fc_pos, "class1_quantile_feasible": RC.feasibility(fc_pos, alpha),
                           "class0_calibration_negatives": int((y["fc"] == 0).sum())}
                    res["runs"].append(rec)
                if abs(alpha - 0.10) < 1e-12:
                    chk = {"model": model, "seed": seed}
                    r = e3[(e3.model == model) & (e3.seed == seed)]
                    st, ec = r[r.method == "static_lac"].iloc[0], r[(r.method == "eci_lac") & (r.lag == 0)].iloc[0]
                    m_st, m_ec = set_metrics(out["static_lac"][0], y_te), set_metrics(out["eci_sets"][0], y_te)
                    chk["static_cov_diff"] = abs(m_st["coverage_marginal"] - st.coverage_marginal)
                    chk["static_size_diff"] = abs(m_st["mean_set_size"] - st.mean_set_size)
                    chk["eci_lr_equal"] = bool(out["eci_sets"][1]["lr"] == ec.lr)
                    chk["eci_cov_diff"] = abs(m_ec["coverage_marginal"] - ec.coverage_marginal)
                    chk["eci_size_diff"] = abs(m_ec["mean_set_size"] - ec.mean_set_size)
                    cf = conf[(conf.model == model) & (conf.seed == seed) & (conf.selected_lr.astype(bool))]
                    m_mo = set_metrics(out["mondrian_eci"][0], y_te)
                    chk["mondrian_lr_equal"] = bool(out["mondrian_eci"][1]["lr"] == cf.iloc[0].lr)
                    chk["mondrian_c1_diff"] = abs(m_mo["coverage_class1"] - cf.iloc[0].eci_coverage_class1)
                    chk["mondrian_size_diff"] = abs(m_mo["mean_set_size"] - cf.iloc[0].eci_mean_set_size)
                    res["checks_at_alpha_0.10"].append(chk)
                    print(model, seed, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in chk.items() if k not in ("model", "seed")}, flush=True)
    ck = pd.DataFrame(res["checks_at_alpha_0.10"])
    res["checks_summary"] = {"max_static_cov_diff": float(ck.static_cov_diff.max()), "max_static_size_diff": float(ck.static_size_diff.max()),
                             "all_eci_lr_equal": bool(ck.eci_lr_equal.all()), "max_eci_cov_diff": float(ck.eci_cov_diff.max()),
                             "max_eci_size_diff": float(ck.eci_size_diff.max()), "all_mondrian_lr_equal": bool(ck.mondrian_lr_equal.all()),
                             "max_mondrian_c1_diff": float(ck.mondrian_c1_diff.max()), "max_mondrian_size_diff": float(ck.mondrian_size_diff.max())}
    print(res["checks_summary"])
    C.dump(res, "phase1", "p1_6_7_operating_and_alpha.json")


if __name__ == "__main__":
    main()
