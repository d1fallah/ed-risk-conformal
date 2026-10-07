"""
Phase 2 sensitivity -- discrimination as a function of the landmark time (L in {0 (arrival), 1, 2, 3, 4, 6} hours after arrival).
Same cohort definitions (R5), split roles, outcome window (24 h), tuning budget (4 configurations, 3 tuning seeds, validation AUROC) and five final seeds as the
primary analysis, for logistic regression and timing-aware gradient-boosted trees (the models with the best and the simplest discrimination; the recurrent models are
not retrained here). L = 0 and L = 2 reuse the primary analysis' predictions. Test AUROC: seed mean, bootstrap 95% interval of the seed mean (1,000 patient-cluster resamples).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402
import p2_1_cohorts as P1  # noqa: E402
import p2_2_classical as P22  # noqa: E402
import m2_build_irregular_cohort as bic  # noqa: E402
from helpers import (convert_temp_to_celcius, remove_outliers, convert_str_to_float, read_triage_table,  # noqa: E402
                     merge_vitalsign_sequence_on_edstay)
from m2_hybrid_common import VITALS_VALID_RANGE  # noqa: E402

HOURS = [1, 3, 4, 6]
VIT = P.VITALS


def seq_for_landmark(df, vs, triage_raw, hours, t_ed):
    bic.T_MAX_ED = t_ed
    bic.T_MAX_TOTAL = 1 + t_ed
    keep = vs[vs["stay_id"].isin(df["stay_id"])].merge(df[["stay_id", "intime"]], on="stay_id")
    keep["charttime"] = pd.to_datetime(keep["charttime"])
    keep = keep[(keep["charttime"] >= keep["intime"]) & (keep["charttime"] <= keep["intime"] + pd.Timedelta(hours=hours))]
    seq = merge_vitalsign_sequence_on_edstay(keep.drop(columns=["intime"]), t_max=t_ed)
    values, obs, pad, delta = bic.build_irregular_sequences(df.reset_index(drop=True), triage_raw, seq)
    th = np.zeros(pad.shape, dtype=np.float32)
    t0 = pd.to_datetime(df["intime"]).values.astype("datetime64[ns]").astype(np.int64)
    sid = df["stay_id"].values
    for i in range(len(df)):
        q = seq.get(sid[i])
        if q is None or len(q) == 0:
            continue
        tt = q["charttime"].values.astype("datetime64[ns]").astype(np.int64)
        k = min(len(tt), t_ed)
        th[i, 1:1 + k] = (tt[:k].astype(np.float64) - t0[i]) / 3.6e12
    return values, obs, pad, th


def main():
    fr = C.load_frames()
    tr, te = fr["train_period"].copy(), fr["test"].copy()
    for k in ("fit", "val", "cs", "fc"):
        tr.loc[tr["stay_id"].isin(fr[k]["stay_id"]), "split_role"] = k
    te["split_role"] = "test"
    fix = pd.read_csv(os.path.join(P.OUTD, "elix_fix.csv")).set_index("stay_id")["count_fixed"]
    for d in (tr, te):
        m = d["stay_id"].isin(fix.index)
        d.loc[m, "comorbidity_elixhauser_count"] = fix.loc[d.loc[m, "stay_id"]].values
    triage_raw = remove_outliers(convert_temp_to_celcius(read_triage_table(os.path.join(bic.MIMIC_ED_PATH, "triage.csv"))), VITALS_VALID_RANGE)
    vs = pd.read_csv(os.path.join(bic.MIMIC_ED_PATH, "vitalsign.csv"))
    vs = vs[vs["stay_id"].isin(np.concatenate([tr["stay_id"].values, te["stay_id"].values]))].copy()
    vs["pain"] = vs["pain"].apply(convert_str_to_float).astype(float)
    vs = vs.rename(columns={c: f"vs_{c}" for c in VIT})
    vs = remove_outliers(convert_temp_to_celcius(vs), VITALS_VALID_RANGE)
    vs = vs.rename(columns={f"vs_{c}": c for c in VIT})
    res = {"hours": [0, 1, 2, 3, 4, 6], "landmarks": {}}
    # primary analysis values for L = 0 and L = 2 (read from the stored contrasts file)
    ct = json.load(open(os.path.join(C.RES, "phase2", "p2_4_contrasts.json")))
    for h, lm in ((0, "triage"), (2, "h2")):
        mm = ct["landmarks"][lm]["models"]
        res["landmarks"][str(h)] = {"source": "primary analysis", "logreg": mm["logreg"]["auroc_seed_mean"], "logreg_ci95": mm["logreg"]["auroc_bootstrap_ci95_of_seed_mean"],
                                    "trees": mm["xgboost_timing" if lm == "h2" else "xgboost"]["auroc_seed_mean"],
                                    "trees_ci95": mm["xgboost_timing" if lm == "h2" else "xgboost"]["auroc_bootstrap_ci95_of_seed_mean"]}
    for h in HOURS:
        tr_l, te_l = P1.label_frame(tr, float(h)), P1.label_frame(te, float(h))
        d = pd.concat([tr_l, te_l])
        d = d[d["lm_eligible"]].copy()
        d["y"] = d["y_primary"].astype(int)
        cnt = None
        # rows within the window -> slot cap covering 99.5% of eligible stays
        v2 = vs.merge(d[["stay_id", "intime"]], on="stay_id")
        v2["charttime"] = pd.to_datetime(v2["charttime"])
        v2 = v2[(v2["charttime"] >= v2["intime"]) & (v2["charttime"] <= v2["intime"] + pd.Timedelta(hours=h))]
        cnt = v2.groupby("stay_id").size().reindex(d["stay_id"].values, fill_value=0)
        t_ed = int(max(2, np.quantile(cnt, 0.995)))
        Xs, ys, dfs = {}, {}, {}
        for role in P.ROLES:
            sub = d[d["split_role"] == role].reset_index(drop=True)
            values, obs, pad, th = seq_for_landmark(sub, vs, triage_raw, h, t_ed)
            agg = P.aggregate_features(values, obs, pad)
            tim = P.timing_features(values, obs, pad, th, L_hours=float(h))
            stat = pd.DataFrame(P.static_matrix(sub), columns=P.STATIC)
            Xs[role] = pd.concat([agg, tim, stat], axis=1)
            ys[role] = sub["y"].values
            dfs[role] = sub
        rec = {"T_ED": t_ed, "eligible": {r: int(len(ys[r])) for r in P.ROLES}, "positives": {r: int(ys[r].sum()) for r in P.ROLES},
               "prevalence_test": float(ys["test"].mean())}
        # logistic regression (deterministic)
        best = max(({"C": c, "val": float(roc_auc_score(ys["val"], P22.lr_model(c).fit(Xs["fit"], ys["fit"]).predict_proba(Xs["val"])[:, 1]))} for c in P22.LR_GRID), key=lambda t: t["val"])
        p_lr = P22.lr_model(best["C"]).fit(Xs["fit"], ys["fit"]).predict_proba(Xs["test"])[:, 1]
        # trees: 4 configurations x 3 tuning seeds
        trace = []
        for depth, lr in P22.XGB_GRID:
            a = []
            for s in C.TUNE_SEEDS:
                m = P22.xgb_model(depth, lr, s).fit(Xs["fit"], ys["fit"], eval_set=[(Xs["val"], ys["val"])], verbose=False)
                a.append(float(roc_auc_score(ys["val"], m.predict_proba(Xs["val"])[:, 1])))
            trace.append({"max_depth": depth, "learning_rate": lr, "val": float(np.mean(a))})
        bx = max(trace, key=lambda t: t["val"])
        p_tr = []
        for s in C.SEEDS:
            m = P22.xgb_model(bx["max_depth"], bx["learning_rate"], s).fit(Xs["fit"], ys["fit"], eval_set=[(Xs["val"], ys["val"])], verbose=False)
            p_tr.append(m.predict_proba(Xs["test"])[:, 1])
        boot = C.ClusterBoot(dfs["test"]["subject_id"].values, B=1000)
        sc_lr = C.Scorer(ys["test"], p_lr)
        sc_tr = [C.Scorer(ys["test"], p) for p in p_tr]
        d_lr = [sc_lr.auroc(w) for w in boot.weights()]
        d_tr = [np.mean([s_.auroc(w) for s_ in sc_tr]) for w in boot.weights()]
        rec.update({"logreg": sc_lr.auroc(), "logreg_ci95": C.pct_ci(d_lr), "logreg_selected": best, "trees": float(np.mean([s_.auroc() for s_ in sc_tr])),
                    "trees_ci95": C.pct_ci(d_tr), "trees_selected": bx, "trees_seed_sd": float(np.std([s_.auroc() for s_ in sc_tr], ddof=1))})
        res["landmarks"][str(h)] = rec
        print(h, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k in ("T_ED", "logreg", "trees", "prevalence_test")}, flush=True)
        C.dump(res, "phase2", "p2_10_landmark_sweep.json")


if __name__ == "__main__":
    main()
