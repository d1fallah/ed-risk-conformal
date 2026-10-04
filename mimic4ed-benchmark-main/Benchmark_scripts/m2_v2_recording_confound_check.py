"""
M2 -- Zero-ED-observation-rate confound check (Phase 3 of the session's
"Clearing the Runway to the Manuscript" plan)
==============================================================================
M2_Findings_Dossier.md Sec 24.1 reports a striking, previously-undocumented
finding: 4.62% of the chronological-primary split's TRAIN period (pre-2016)
stays have zero recorded ED-vitalsign observations, vs. only 0.02% of the
TEST period (post-2016) -- a 231x difference straddling the same 2016 cut
that defines the outer train/test boundary. Sec 24.5 item 6 flags this as
needing its own check: does this recording-practice shift confound any of
Sec 24.2's discrimination deltas, rather than reflecting a genuine
representation improvement?

This is a real risk because "zero ED observations" means every v2 model
falls back to triage-only input for that row (obs_mask all-zero past
t=0) -- if the TEST period's near-total elimination of that degraded case
is itself responsible for part of the v2-vs-T=2 AUROC gain (rather than
the richer sequence on stays that DO have real ED observations), that
would be a confound worth disclosing prominently, not folding silently
into "the rebuild works."

Method: stratify TEST-period predictions (already saved,
data/m2_v2_predictions.csv) by has_ed_observations (from the persisted
m2_irregular_sequences_test_v2.npz pad_mask, t=1 onward), and compare
each v2 model's AUROC within each stratum. Since only 17 of 84,241 test
rows lack ED observations (0.02%, Sec 24.1), this is a severe class-size
imbalance for the confound check itself -- reported honestly, not
smoothed over.

No retraining: reads already-saved predictions and the already-built
sequence tensors.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_PATH = os.path.join(DATA_DIR, "m2_v2_recording_confound_check.json")

sys.path.insert(0, SCRIPT_DIR)
from m2_v2_common import load_split_frame


def main():
    df_test = load_split_frame("test_v2")
    with np.load(os.path.join(DATA_DIR, "m2_irregular_sequences_test_v2.npz")) as z:
        pad_mask = z["pad_mask"]  # (N, 11); t=0 is always 1 (triage); t=1.. real ED obs

    has_ed_obs = pad_mask[:, 1:].any(axis=1)  # True if >=1 real ED-vitalsign observation
    n_no_ed = int((~has_ed_obs).sum())
    n_with_ed = int(has_ed_obs.sum())
    print(f"Test period: {n_with_ed} stays with >=1 ED observation, {n_no_ed} with none "
          f"({n_no_ed/len(df_test):.4%})")

    # Map stay_id -> has_ed_obs (df_test and pad_mask share row order, per
    # m2_v2_common.py's own alignment contract)
    stay_to_has_ed = dict(zip(df_test["stay_id"].values, has_ed_obs))

    preds = pd.read_csv(os.path.join(DATA_DIR, "m2_v2_predictions.csv"))
    preds["has_ed_obs"] = preds["stay_id"].map(stay_to_has_ed)
    missing_map = preds["has_ed_obs"].isna().sum()
    if missing_map:
        print(f"WARNING: {missing_map} prediction rows had no stay_id match in test_v2 pad_mask -- excluded.")
        preds = preds.dropna(subset=["has_ed_obs"])

    results = {"n_with_ed_obs": n_with_ed, "n_no_ed_obs": n_no_ed,
               "no_ed_obs_fraction": n_no_ed / len(df_test), "by_model": {}}

    for model in sorted(preds["model"].unique()):
        sub = preds[preds["model"] == model]
        per_seed_full, per_seed_with_ed, per_seed_no_ed = [], [], []
        for seed in sorted(sub["seed"].unique()):
            s = sub[sub["seed"] == seed]
            s_with = s[s["has_ed_obs"]]
            s_no = s[~s["has_ed_obs"]]
            if s["y_true"].nunique() > 1:
                per_seed_full.append(roc_auc_score(s["y_true"], s["y_prob"]))
            if len(s_with) and s_with["y_true"].nunique() > 1:
                per_seed_with_ed.append(roc_auc_score(s_with["y_true"], s_with["y_prob"]))
            if len(s_no) and s_no["y_true"].nunique() > 1:
                per_seed_no_ed.append(roc_auc_score(s_no["y_true"], s_no["y_prob"]))
            else:
                print(f"  {model} seed={seed}: no-ED-obs stratum too small/degenerate for AUROC "
                      f"(n={len(s_no)}, pos={int(s_no['y_true'].sum()) if len(s_no) else 0})")

        entry = {
            "auroc_full_mean": float(np.mean(per_seed_full)) if per_seed_full else None,
            "auroc_with_ed_obs_mean": float(np.mean(per_seed_with_ed)) if per_seed_with_ed else None,
            "auroc_no_ed_obs_mean": float(np.mean(per_seed_no_ed)) if per_seed_no_ed else None,
            "n_seeds_no_ed_obs_computable": len(per_seed_no_ed),
        }
        results["by_model"][model] = entry
        print(f"{model:35s} full={entry['auroc_full_mean']}  "
              f"with_ED={entry['auroc_with_ed_obs_mean']}  "
              f"no_ED={entry['auroc_no_ed_obs_mean']} (n_seeds={entry['n_seeds_no_ed_obs_computable']}/5)")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")
    print(
        "\nInterpretation note: the no-ED-obs stratum has only 17 test rows (Sec 24.1), too few for "
        "a stable AUROC or bootstrap CI on its own -- this check can show whether the WITH-ED-obs "
        "stratum alone still reproduces Sec 24.2's discrimination gains (the informative comparison), "
        "not draw a reliable conclusion about the no-ED-obs stratum specifically."
    )


if __name__ == "__main__":
    main()
