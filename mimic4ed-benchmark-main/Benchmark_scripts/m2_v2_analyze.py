"""
M2 -- Patient-Cluster Bootstrap for the v2 (irregular-series) Deltas
==============================================================================
Phase 1 of the session's "Clearing the Runway to the Manuscript" plan: the
single hardest blocker before any v2 comparison is reportable. Every
number in M2_Findings_Dossier.md Sec 24 is currently a point estimate +/-
seed-SD only -- no inferential CI. Sec 24.4's own framing note already
downgrades the headline AUROC comparison to "suggestive, not conclusive"
on that basis alone.

Reuses m2_hybrid_analyze.py::patient_cluster_bootstrap() UNCHANGED (same
resampling unit: patients resampled with replacement, all of a sampled
patient's encounters kept with multiplicity, never deduplicated) -- this
is not a new statistical procedure, it is the same one already validated
and reported on the T=2 cohort, applied to v2's saved predictions. No
retraining: reads data/m2_v2_predictions.csv (long format: subject_id,
stay_id, model, seed, y_true, y_prob).

Five v2 deltas (mirrors the T=2 set; Delta_rich dropped -- no v2
XGBoost-Rich exists or is planned; Delta_modern_comparator added, since
mTAN v2 is this session's new comparator and Sec 24.5 item 5 names the
missing CI as the most manuscript-blocking gap):

  Delta_representation      = Hybrid GRU-D v2 - XGBoost-Matched v2      (PRIMARY)
  Delta_static_contribution = Hybrid GRU-D v2 - Frozen-Temporal Head Control v2
  Delta_temporal_alone      = Temporal-Only GRU-D v2 - XGBoost-Matched v2
  Delta_static_only         = Static-Only MLP v2 - Temporal-Only GRU-D v2
  Delta_modern_comparator   = Hybrid GRU-D v2 - mTAN v2

N_BOOTSTRAP=10000, matching the raised value m2_matched_tuning_sensitivity.py
already adopted after Sec 19.2's Monte-Carlo-stability finding (the T=2
default of 1000 was shown there to be unstable for a lower-CI-bound
decision). Reports point estimate (mean across seeds), SD across seeds
(training-seed variability), and the 95% patient-cluster bootstrap CI
(population uncertainty) as three explicitly distinct quantities, per
Sec 18's own convention -- never conflated.

Holm correction applied across the declared 5-delta confirmatory family,
per Sec 22 finding #10 ("~11 confirmatory tests... unadjusted for
multiplicity" -- disclosed as a Should-fix there; applied by construction
here rather than repeating the gap on the v2 family).
"""
import os, sys, json, argparse, warnings
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "m2_v2_predictions.csv")
OUT_PATH = os.path.join(DATA_DIR, "m2_v2_deltas.json")
N_BOOTSTRAP = 10000
RNG_SEED = 42
FINAL_SEEDS = [42, 123, 7, 456, 789]

# --reuse-from lets a caller skip recomputing a delta whose two models' saved
# predictions provably did not change since a prior run (e.g. after Phase
# 2's tuning-symmetry fix only touched 3 of the 6 v2 models) -- the skipped
# delta's bootstrap result is copied byte-for-byte from the prior JSON
# rather than re-run, since re-running on identical inputs with the same
# RNG seed would reproduce the identical number at real compute cost. This
# is disabled (every delta always recomputed) unless explicitly requested,
# so the reproducibility entrypoint's default `analyze` stage still
# recomputes every delta from scratch.

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_analyze import patient_cluster_bootstrap  # reused unchanged

DELTAS = [
    ("Delta_representation", "Hybrid GRU-D v2", "XGBoost-Matched v2",
     "PRIMARY: does temporal representation add value over matched tabular aggregation, on the irregular-series cohort?"),
    ("Delta_static_contribution", "Hybrid GRU-D v2", "Frozen-Temporal Head Control v2",
     "Cleanest ablation: incremental contribution of the static MLP branch alone."),
    ("Delta_temporal_alone", "Temporal-Only GRU-D v2", "XGBoost-Matched v2",
     "How strong is temporal representation alone vs. matched tabular aggregation?"),
    ("Delta_static_only", "Static-Only MLP v2", "Temporal-Only GRU-D v2",
     "Relative contribution of static characteristics vs. temporal representation."),
    ("Delta_modern_comparator", "Hybrid GRU-D v2", "mTAN v2",
     "Does GRU-D's decay-based recurrence beat a tuned attention-based comparator on this irregular series?"),
]


def wide_for_seed(preds, seed, model_a, model_b):
    """Pivots the long-format v2 predictions file to the (subject_id,
    stay_id, y_true, prob_a, prob_b) wide shape patient_cluster_bootstrap()
    expects, for one seed and one pair of models."""
    da = preds[(preds["seed"] == seed) & (preds["model"] == model_a)][
        ["subject_id", "stay_id", "y_true", "y_prob"]].rename(columns={"y_prob": "prob_a"})
    db = preds[(preds["seed"] == seed) & (preds["model"] == model_b)][
        ["stay_id", "y_prob"]].rename(columns={"y_prob": "prob_b"})
    merged = da.merge(db, on="stay_id", how="inner")
    assert len(merged) == len(da) == len(db), (
        f"seed={seed} {model_a} vs {model_b}: row count mismatch after merge "
        f"({len(da)} / {len(db)} / {len(merged)}) -- the two models were not "
        f"scored on the same test rows. STOP."
    )
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-from", default=None,
                         help="prior m2_v2_deltas.json to copy unchanged deltas from")
    parser.add_argument("--skip", nargs="*", default=[],
                         help="delta names to reuse from --reuse-from instead of recomputing "
                              "(only valid when neither model's saved predictions changed)")
    args = parser.parse_args()

    reuse_source = None
    if args.skip:
        assert args.reuse_from, "--skip requires --reuse-from"
        with open(args.reuse_from) as f:
            reuse_source = json.load(f)
        for name in args.skip:
            assert name in reuse_source, f"--skip {name} not found in {args.reuse_from}"
        print(f"Reusing {args.skip} verbatim from {args.reuse_from} (unchanged model predictions); "
              f"recomputing the rest.")

    preds = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded {PREDICTIONS_PATH}: {len(preds)} rows, "
          f"{preds['model'].nunique()} models, {preds['seed'].nunique()} seeds")
    print("Models present:", sorted(preds["model"].unique()))

    results = {}
    pvalue_proxies = {}  # delta_name -> smallest-seed-CI-based two-sided "CI excludes zero" proxy p

    for delta_name, model_a, model_b, question in DELTAS:
        if delta_name in args.skip:
            prior = reuse_source[delta_name]
            assert prior["model_a"] == model_a and prior["model_b"] == model_b, (
                f"{delta_name}: --reuse-from model pair {prior['model_a']}/{prior['model_b']} "
                f"does not match current DELTAS definition {model_a}/{model_b} -- refusing to reuse."
            )
            results[delta_name] = {k: v for k, v in prior.items()
                                    if k not in ("holm_corrected_p", "holm_significant_at_0.05")}
            pvalue_proxies[delta_name] = prior["p_proxy_conservative"]
            print(f"\n{'='*60}\n{delta_name}: {model_a} - {model_b}  [REUSED, not recomputed]\n"
                  f"mean={prior['mean']:+.4f} +/- SD={prior['sd']:.4f}\n{'='*60}")
            continue
        print(f"\n{'='*60}\n{delta_name}: {model_a} - {model_b}\n{question}\n{'='*60}")
        per_seed = []
        for seed in FINAL_SEEDS:
            df_seed = wide_for_seed(preds, seed, model_a, model_b)
            point, ci_lo, ci_hi, boot = patient_cluster_bootstrap(
                df_seed, "subject_id", "prob_a", "prob_b", "y_true",
                n_boot=N_BOOTSTRAP, seed=RNG_SEED + seed,
            )
            # two-sided bootstrap p-value proxy: 2 * min(P(delta<=0), P(delta>=0))
            # under the resampled null-adjacent distribution -- standard
            # percentile-bootstrap p-value construction, used here only to
            # feed the Holm correction below (the CI itself remains the
            # primary reported quantity).
            p_below = float((boot <= 0).mean())
            p_above = float((boot >= 0).mean())
            p_proxy = min(1.0, 2 * min(p_below, p_above))
            per_seed.append({"seed": seed, "point_estimate": point, "ci_lo": ci_lo, "ci_hi": ci_hi,
                              "excludes_zero": bool(ci_lo > 0 or ci_hi < 0), "p_proxy": p_proxy})
            print(f"  seed={seed}: {delta_name} = {point:+.4f}  bootstrap 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]"
                  f"  excludes_zero={ci_lo > 0 or ci_hi < 0}  p_proxy={p_proxy:.4f}")
        points = np.array([r["point_estimate"] for r in per_seed])
        # Family-level p-value proxy: the LARGEST (most conservative) of the
        # 5 per-seed proxies, so one lucky seed cannot carry the whole
        # delta's significance claim.
        family_p = max(r["p_proxy"] for r in per_seed)
        pvalue_proxies[delta_name] = family_p
        print(f"  Cross-seed: mean={points.mean():+.4f} +/- SD={points.std(ddof=1):.4f} "
              f"(training-seed variability, distinct from the bootstrap CI's population uncertainty)")
        results[delta_name] = {
            "model_a": model_a, "model_b": model_b, "question": question,
            "per_seed": per_seed, "mean": float(points.mean()), "sd": float(points.std(ddof=1)),
            "p_proxy_conservative": family_p,
            "excludes_zero_at_n_of_5_seeds": sum(r["excludes_zero"] for r in per_seed),
        }

    # Holm step-down correction across the 5-delta confirmatory family
    # (Sec 22 finding #10), applied to each delta's conservative p-proxy.
    names = list(pvalue_proxies.keys())
    pvals = [pvalue_proxies[n] for n in names]
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    print(f"\n{'='*60}\nHolm correction across the {len(names)}-delta confirmatory family\n{'='*60}")
    for name, raw_p, corr_p, rej in zip(names, pvals, pvals_corrected, reject):
        results[name]["holm_corrected_p"] = float(corr_p)
        results[name]["holm_significant_at_0.05"] = bool(rej)
        print(f"  {name}: raw p_proxy={raw_p:.4f}  Holm-corrected={corr_p:.4f}  "
              f"significant={rej}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
