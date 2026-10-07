"""
Phase 2 supplementary analysis (tests the explanation offered for the fall in AUROC at later landmarks).
Hypothesis stated before running: the eligible population changes with the landmark (stays that ended or had an event drop out). If the later-landmark decline is mainly a
population effect, then on ONE fixed set of test stays (those still in the ED and event-free at 6 hours, hence eligible at every earlier landmark) AUROC should not fall as the
landmark moves later; if it falls, the decline is not explained by the changing population alone.
Protocol: identical to p2_10 (same cohort definitions, split roles, features, equal tuning budget, five seeds) for logistic regression and the timing-aware trees at
L = 1, 3, 4, 6; per-stay test predictions are kept, L = 2 predictions are the stored primary-analysis predictions. Each landmark is scored on its own outcome window
(L, L + 24 h] on the fixed set; the fixed set's event rate therefore differs by landmark and is reported. As a check, the AUROCs on each landmark's own eligible cohort
are compared with the stored values of p2_10 (they should agree closely; deterministic logistic regression exactly).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402
import p2_1_cohorts as P1  # noqa: E402
import p2_2_classical as P22  # noqa: E402
import p2_10_landmark_sweep as SW  # noqa: E402
import m2_build_irregular_cohort as bic  # noqa: E402
from helpers import convert_temp_to_celcius, remove_outliers, convert_str_to_float, read_triage_table  # noqa: E402
from m2_hybrid_common import VITALS_VALID_RANGE  # noqa: E402

HOURS = [1, 3, 4, 6]
VIT = P.VITALS


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
    stored = json.load(open(os.path.join(C.RES, "phase2", "p2_10_landmark_sweep.json")))
    # fixed set: test stays eligible at 6 h
    te6 = P1.label_frame(te, 6.0)
    fixed_ids = set(te6.loc[te6["lm_eligible"], "stay_id"].values)
    res = {"fixed_set_size": len(fixed_ids), "hours": [1, 2, 3, 4, 6], "landmarks": {}}
    preds = {}
    # L = 2 : stored primary-analysis predictions
    df2 = P.load_cohort("h2", "primary")
    te2 = P.role(df2, "test")
    pl = {}
    sid, sc = P.load_preds("h2_logreg_test", 42)
    pl["logreg"] = (sid, np.asarray(sc, float))
    arr = []
    for s in C.SEEDS:
        sid, sc = P.load_preds("h2_xgboost_timing_test", s)
        arr.append(np.asarray(sc, float))
    pl["trees"] = (sid, np.mean(arr, axis=0))
    preds[2] = {"stay_id": sid, "y": te2["y"].values.astype(int), "pid": te2["subject_id"].values, "scores": pl}
    for h in HOURS:
        tr_l, te_l = P1.label_frame(tr, float(h)), P1.label_frame(te, float(h))
        d = pd.concat([tr_l, te_l])
        d = d[d["lm_eligible"]].copy()
        d["y"] = d["y_primary"].astype(int)
        v2 = vs.merge(d[["stay_id", "intime"]], on="stay_id")
        v2["charttime"] = pd.to_datetime(v2["charttime"])
        v2 = v2[(v2["charttime"] >= v2["intime"]) & (v2["charttime"] <= v2["intime"] + pd.Timedelta(hours=h))]
        cnt = v2.groupby("stay_id").size().reindex(d["stay_id"].values, fill_value=0)
        t_ed = int(max(2, np.quantile(cnt, 0.995)))
        Xs, ys, dfs = {}, {}, {}
        for role in P.ROLES:
            sub = d[d["split_role"] == role].reset_index(drop=True)
            values, obs, pad, th = SW.seq_for_landmark(sub, vs, triage_raw, h, t_ed)
            agg = P.aggregate_features(values, obs, pad)
            tim = P.timing_features(values, obs, pad, th, L_hours=float(h))
            stat = pd.DataFrame(P.static_matrix(sub), columns=P.STATIC)
            Xs[role] = pd.concat([agg, tim, stat], axis=1)
            ys[role] = sub["y"].values
            dfs[role] = sub
        best = max(({"C": c, "val": float(roc_auc_score(ys["val"], P22.lr_model(c).fit(Xs["fit"], ys["fit"]).predict_proba(Xs["val"])[:, 1]))} for c in P22.LR_GRID), key=lambda t: t["val"])
        p_lr = P22.lr_model(best["C"]).fit(Xs["fit"], ys["fit"]).predict_proba(Xs["test"])[:, 1]
        trace = []
        for depth, lr in P22.XGB_GRID:
            a = [float(roc_auc_score(ys["val"], P22.xgb_model(depth, lr, s).fit(Xs["fit"], ys["fit"], eval_set=[(Xs["val"], ys["val"])], verbose=False).predict_proba(Xs["val"])[:, 1])) for s in C.TUNE_SEEDS]
            trace.append({"max_depth": depth, "learning_rate": lr, "val": float(np.mean(a))})
        bx = max(trace, key=lambda t: t["val"])
        p_tr = np.mean([P22.xgb_model(bx["max_depth"], bx["learning_rate"], s).fit(Xs["fit"], ys["fit"], eval_set=[(Xs["val"], ys["val"])], verbose=False).predict_proba(Xs["test"])[:, 1]
                        for s in C.SEEDS], axis=0)
        preds[h] = {"stay_id": dfs["test"]["stay_id"].values, "y": ys["test"], "pid": dfs["test"]["subject_id"].values, "scores": {"logreg": (dfs["test"]["stay_id"].values, p_lr), "trees": (dfs["test"]["stay_id"].values, p_tr)}}
        print("trained landmark", h, flush=True)
    for h in [1, 2, 3, 4, 6]:
        p = preds[h]
        rec = {"own_cohort": {"n": int(len(p["y"])), "prevalence": float(p["y"].mean())}, "fixed_set": {}}
        m = np.isin(p["stay_id"], list(fixed_ids))
        rec["fixed_set"] = {"n": int(m.sum()), "positives": int(p["y"][m].sum()), "prevalence": float(p["y"][m].mean())}
        boot = C.ClusterBoot(p["pid"][m], B=1000)
        for k in ("logreg", "trees"):
            s_all = p["scores"][k][1]
            rec["own_cohort"][k] = float(roc_auc_score(p["y"], s_all))
            sc = C.Scorer(p["y"][m], s_all[m])
            rec["fixed_set"][k] = sc.auroc()
            rec["fixed_set"][k + "_ci95"] = C.pct_ci([sc.auroc(w) for w in boot.weights()])
            ref = stored["landmarks"][str(h)][k]
            rec["own_cohort"][k + "_stored_p2_10"] = float(ref)
        res["landmarks"][str(h)] = rec
        print(h, rec["fixed_set"], {k: (round(rec["own_cohort"][k], 4), round(rec["own_cohort"][k + "_stored_p2_10"], 4)) for k in ("logreg", "trees")}, flush=True)
    C.dump(res, "phase2", "p2_17_fixed_cohort_landmarks.json")


if __name__ == "__main__":
    main()
