"""
Phase 2.3 analysis -- discrimination and the pre-specified AUROC contrasts (R7).

Test-period scores of every Phase 2 model are scored on the primary outcome Y_L (and, as a sensitivity, on the secondary episode-endpoint
outcome restricted to eligible stays with t_event > L). Uncertainty is reported in two separate layers (rule 10):
  * seed variability: mean +- SD over the five training seeds;
  * sampling uncertainty: patient-cluster bootstrap (patients resampled with all their stays), 10,000 resamples for the primary-outcome
    AUROC and contrasts, 1,000 for AUPRC and the secondary outcome; paired draws across models and seeds.
Contrasts (Holm over all five; significant only if Holm-significant AND every per-seed 95% CI excludes zero):
  2 h:  C1 Hybrid-E2E - timing-aware XGBoost;  C2 Hybrid-E2E - logistic regression;  C3 Hybrid-E2E - Temporal-Only GRU-D;
        C4 Hybrid-E2E - Hybrid-frozen
  triage: C5 XGBoost - logistic regression
Per-seed contrasts pair seed s of both models; the deterministic logistic regression is compared with each seed of the other model.
The Holm p-value of a contrast comes from the bootstrap distribution of the seed-averaged difference.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402

B_MAIN, B_SEC = 10000, 1000
MODELS = {
    "h2": [("logreg", "h2_logreg", [42]), ("xgboost_timing", "h2_xgboost_timing", C.SEEDS), ("temporal_grud", "h2_temporal_grud", C.SEEDS),
           ("hybrid_frozen", "h2_hybrid_frozen", C.SEEDS), ("hybrid_e2e", "h2_hybrid_e2e", C.SEEDS)],
    "triage": [("logreg", "triage_logreg", [42]), ("xgboost", "triage_xgboost", C.SEEDS)],
}
CONTRASTS = [("C1", "h2", "hybrid_e2e", "xgboost_timing"), ("C2", "h2", "hybrid_e2e", "logreg"), ("C3", "h2", "hybrid_e2e", "temporal_grud"),
             ("C4", "h2", "hybrid_e2e", "hybrid_frozen"), ("C5", "triage", "xgboost", "logreg")]


def scores_for(lm, key, seeds, df_test):
    out = {}
    for s in seeds:
        sid, sc = P.load_preds(key, s) if False else P.load_preds(f"{key}_test", s)
        assert np.array_equal(sid, df_test["stay_id"].values), f"{key} seed {s}: stay order differs from the cohort"
        out[s] = sc
    return out


def main():
    res = {"bootstrap_primary_resamples": B_MAIN, "bootstrap_secondary_resamples": B_SEC, "landmarks": {}}
    for lm in ("h2", "triage"):
        df = P.load_cohort(lm, "primary")
        te = P.role(df, "test")
        y = te["y"].values.astype(int)
        sec = P.load_cohort(lm, "secondary")
        te2 = P.role(sec, "test")
        keep2 = te["stay_id"].isin(te2["stay_id"]).values
        y2_map = te2.set_index("stay_id")["y"]
        y2 = y2_map.loc[te["stay_id"].values[keep2]].values.astype(int)
        boot = C.ClusterBoot(te["subject_id"].values, B=B_MAIN)
        S = {name: scores_for(lm, key, seeds, te) for name, key, seeds in MODELS[lm]}
        sc = {(n, s): C.Scorer(y, v) for n, d in S.items() for s, v in d.items()}
        point = {k: v.auroc() for k, v in sc.items()}
        point_prc = {k: v.auprc() for k, v in sc.items()}
        draws = {k: np.empty(B_MAIN) for k in sc}
        for b, w in enumerate(boot.weights()):
            for k, v in sc.items():
                draws[k][b] = v.auroc(w)
        rec = {"n_test": int(len(y)), "positives": int(y.sum()), "models": {}, "contrasts": {}}
        for name, _key, seeds in MODELS[lm]:
            per = [point[(name, s)] for s in seeds]
            seed_mean_draws = np.mean([draws[(name, s)] for s in seeds], axis=0)
            rec["models"][name] = {"seeds": seeds, "auroc_per_seed": per, "auroc_seed_mean": float(np.mean(per)),
                                   "auroc_seed_sd": float(np.std(per, ddof=1)) if len(per) > 1 else 0.0,
                                   "auroc_bootstrap_ci95_per_seed": [C.pct_ci(draws[(name, s)]) for s in seeds],
                                   "auroc_bootstrap_ci95_of_seed_mean": C.pct_ci(seed_mean_draws),
                                   "auprc_per_seed": [point_prc[(name, s)] for s in seeds],
                                   "auprc_seed_mean": float(np.mean([point_prc[(name, s)] for s in seeds])),
                                   "auprc_seed_sd": float(np.std([point_prc[(name, s)] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0}
        # AUPRC bootstrap (1,000) for the seed mean
        boot_s = C.ClusterBoot(te["subject_id"].values, B=B_SEC)
        prc_draws = {k: np.empty(B_SEC) for k in sc}
        for b, w in enumerate(boot_s.weights()):
            for k, v in sc.items():
                prc_draws[k][b] = v.auprc(w)
        for name, _key, seeds in MODELS[lm]:
            rec["models"][name]["auprc_bootstrap_ci95_of_seed_mean"] = C.pct_ci(np.mean([prc_draws[(name, s)] for s in seeds], axis=0))
        # contrasts
        for cid, lmc, a, bname in CONTRASTS:
            if lmc != lm:
                continue
            seeds = C.SEEDS
            per_seed = {}
            deltas = []
            for s in seeds:
                sb = 42 if bname == "logreg" else s
                d = draws[(a, s)] - draws[(bname, sb)]
                deltas.append(d)
                per_seed[s] = {"delta": point[(a, s)] - point[(bname, sb)], "ci95": C.pct_ci(d),
                               "excludes_zero": bool(C.pct_ci(d)[0] > 0 or C.pct_ci(d)[1] < 0)}
            mean_d = np.mean(deltas, axis=0)
            rec["contrasts"][cid] = {"a": a, "b": bname, "per_seed": per_seed,
                                     "delta_seed_mean": float(np.mean([v["delta"] for v in per_seed.values()])),
                                     "delta_seed_sd": float(np.std([v["delta"] for v in per_seed.values()], ddof=1)),
                                     "delta_bootstrap_ci95_of_seed_mean": C.pct_ci(mean_d),
                                     "p_boot_two_sided": C.boot_p_two_sided(mean_d), "all_seed_cis_exclude_zero": all(v["excludes_zero"] for v in per_seed.values())}
        # secondary outcome (sensitivity)
        sec_sc = {k: C.Scorer(y2, S[k[0]][k[1]][keep2]) for k in sc}
        boot2 = C.ClusterBoot(te["subject_id"].values[keep2], B=B_SEC)
        d2 = {k: np.empty(B_SEC) for k in sec_sc}
        for b, w in enumerate(boot2.weights()):
            for k, v in sec_sc.items():
                d2[k][b] = v.auroc(w)
        rec["secondary_outcome"] = {"n_test": int(len(y2)), "positives": int(y2.sum()),
                                    "models": {n: {"auroc_seed_mean": float(np.mean([sec_sc[(n, s)].auroc() for s in seeds])),
                                                   "auroc_seed_sd": float(np.std([sec_sc[(n, s)].auroc() for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0,
                                                   "bootstrap_ci95_of_seed_mean": C.pct_ci(np.mean([d2[(n, s)] for s in seeds], axis=0))}
                                               for n, _k, seeds in MODELS[lm]}}
        res["landmarks"][lm] = rec
        print(lm, {n: round(v["auroc_seed_mean"], 4) for n, v in rec["models"].items()}, flush=True)
    # Holm over all five
    ids = [c[0] for c in CONTRASTS]
    allc = {cid: res["landmarks"][lm]["contrasts"][cid] for cid, lm, _a, _b in CONTRASTS}
    adj = C.holm([allc[c]["p_boot_two_sided"] for c in ids])
    for cid, a in zip(ids, adj):
        allc[cid]["holm_adjusted_p"] = a
        allc[cid]["significant"] = bool(a < 0.05 and allc[cid]["all_seed_cis_exclude_zero"])
    res["holm_over"] = ids
    res["significance_rule"] = "Holm-adjusted p < 0.05 AND all five per-seed 95% CIs exclude zero"
    C.dump(res, os.environ.get("P2_PHASE", "phase2"), "p2_4_contrasts.json")
    for cid in ids:
        c = allc[cid]
        print(cid, c["a"], "-", c["b"], round(c["delta_seed_mean"], 4), c["delta_bootstrap_ci95_of_seed_mean"], "holm", round(c["holm_adjusted_p"], 4), "sig", c["significant"])


if __name__ == "__main__":
    main()
