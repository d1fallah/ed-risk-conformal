"""
Phase 2.2 / 2.3 (classical models) -- logistic regression and XGBoost at the triage landmark; logistic regression and timing-aware XGBoost at the
2 h landmark. Tuning budget as fixed in results/revision_v2/phase2/TUNING_BUDGET.md (4 configurations, tuning seeds {42, 123, 7}, validation AUROC).
Final refit on `training` for five seeds; scores for every role are stored in data/revision_v2/preds/ (not printed).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402

LR_GRID = [0.01, 0.1, 1.0, 10.0]
XGB_GRID = [(4, 0.05), (6, 0.05), (4, 0.1), (6, 0.1)]


def lr_model(c):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def xgb_model(depth, lr, seed):
    return xgb.XGBClassifier(n_estimators=1000, max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                             reg_lambda=1.0, eval_metric="auc", early_stopping_rounds=50, tree_method="hist", n_jobs=4, random_state=seed,
                             verbosity=0)


def run_landmark(lm, feats, xgb_name, out):
    df = P.load_cohort(lm, "primary")
    R = {r: P.role(df, r) for r in P.ROLES}
    X = {r: feats(R[r]) for r in P.ROLES}
    y = {r: R[r]["y"].values for r in P.ROLES}
    rec = {"landmark": lm, "n": {r: int(len(R[r])) for r in P.ROLES}, "positives": {r: int(y[r].sum()) for r in P.ROLES},
           "n_features": int(X["fit"].shape[1]), "features": list(X["fit"].columns)}
    # ---- logistic regression (deterministic)
    trace = []
    for c in LR_GRID:
        m = lr_model(c).fit(X["fit"], y["fit"])
        trace.append({"C": c, "val_auroc": float(roc_auc_score(y["val"], m.predict_proba(X["val"])[:, 1]))})
    best = max(trace, key=lambda t: t["val_auroc"])      # max returns the first maximum -> listed order is the tie-break
    m = lr_model(best["C"]).fit(X["fit"], y["fit"])
    for r in P.ROLES:
        P.save_preds(f"{lm}_logreg_{r}", 42, R[r], m.predict_proba(X[r])[:, 1])
    rec["logreg"] = {"tuning_trace": trace, "selected": best, "deterministic_single_fit": True}
    print(lm, "logreg selected", best, flush=True)
    # ---- XGBoost
    trace = []
    for depth, lr in XGB_GRID:
        aucs = []
        for s in C.TUNE_SEEDS:
            mm = xgb_model(depth, lr, s)
            mm.fit(X["fit"], y["fit"], eval_set=[(X["val"], y["val"])], verbose=False)
            aucs.append(float(roc_auc_score(y["val"], mm.predict_proba(X["val"])[:, 1])))
        trace.append({"max_depth": depth, "learning_rate": lr, "val_auroc_per_tune_seed": aucs, "val_auroc_mean": float(np.mean(aucs))})
        print(lm, xgb_name, depth, lr, round(float(np.mean(aucs)), 4), flush=True)
    best = max(trace, key=lambda t: t["val_auroc_mean"])
    finals = {}
    for s in C.SEEDS:
        mm = xgb_model(best["max_depth"], best["learning_rate"], s)
        mm.fit(X["fit"], y["fit"], eval_set=[(X["val"], y["val"])], verbose=False)
        for r in P.ROLES:
            P.save_preds(f"{lm}_{xgb_name}_{r}", s, R[r], mm.predict_proba(X[r])[:, 1])
        finals[s] = {"best_iteration": int(mm.best_iteration), "val_auroc": float(roc_auc_score(y["val"], mm.predict_proba(X["val"])[:, 1]))}
    rec[xgb_name] = {"tuning_trace": trace, "selected": {k: best[k] for k in ("max_depth", "learning_rate")}, "finals": finals}
    out[lm] = rec


def main():
    out = {"budget": "TUNING_BUDGET.md", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    run_landmark("triage", P.triage_feature_frame, "xgboost", out)
    run_landmark("h2", lambda d: P.h2_feature_frame(d, timing=True), "xgboost_timing", out)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.dump(out, "phase2", "p2_2_classical.json")


if __name__ == "__main__":
    main()
