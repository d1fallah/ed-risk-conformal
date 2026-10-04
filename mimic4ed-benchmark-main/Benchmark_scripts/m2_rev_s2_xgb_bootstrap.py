"""
M2 revision S2 -- uncertainty for the deterministic tree model
==============================================================================
Executes S2 of M2_Revision_PreSpecification.md (supervisor sign-off 2026-10-03).

XGBoost-Matched v2 is deterministic: its five seed runs give identical predictions, so its seed SD
is exactly zero. Its AUROC and AUPRC cells report instead the 95% patient-cluster bootstrap CI
(10,000 resamples; patients drawn with replacement with all their stays, as in
m2_hybrid_analyze.patient_cluster_bootstrap; RNG seed 42). This is a population-uncertainty
interval, a different quantity from the neural models' seed SD, and is labelled as such.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_JSON = os.path.join(DATA_DIR, "m2_v2_rev_s2_xgb_bootstrap.json")
MODEL = "XGBoost-Matched v2"
N_BOOT = 10000
RNG_SEED = 42


def main():
    preds = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_predictions.csv"))
    d = preds[preds.model == MODEL]
    seeds = sorted(int(x) for x in d.seed.unique())
    base = d[d.seed == seeds[0]].sort_values("stay_id").reset_index(drop=True)
    for s in seeds[1:]:
        other = d[d.seed == s].sort_values("stay_id").reset_index(drop=True)
        assert (other.stay_id.values == base.stay_id.values).all()
        assert np.array_equal(other.y_prob.values, base.y_prob.values), f"seed {s} differs: model is not deterministic"
    print(f"{MODEL}: predictions identical across seeds {seeds}")

    y = base.y_true.values.astype(int); p = base.y_prob.values
    point = {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}
    by_patient = {k: v.values for k, v in base.groupby("subject_id").groups.items()}
    patients = np.array(list(by_patient.keys()))
    rng = np.random.RandomState(RNG_SEED)
    boot = {"auroc": np.empty(N_BOOT), "auprc": np.empty(N_BOOT)}
    for b in range(N_BOOT):
        idx = np.concatenate([by_patient[q] for q in rng.choice(patients, size=len(patients), replace=True)])
        boot["auroc"][b] = roc_auc_score(y[idx], p[idx])
        boot["auprc"][b] = average_precision_score(y[idx], p[idx])
    out = {"model": MODEL, "seeds_identical": seeds, "n_stays": int(len(base)), "n_patients": int(len(patients)),
           "n_boot": N_BOOT, "rng_seed": RNG_SEED}
    for k in ("auroc", "auprc"):
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out[k] = {"point": point[k], "ci_lo": float(lo), "ci_hi": float(hi)}
        print(f"{k}: {point[k]:.4f} [{lo:.4f}, {hi:.4f}]")
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
