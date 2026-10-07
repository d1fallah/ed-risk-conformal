"""
Phase 1.3 -- Elixhauser ablation (Phase 0 found the count valid by design, so this is the planned sensitivity analysis).

Retrains, WITHOUT comorbidity_elixhauser_count:
  * the Hybrid GRU-D static branch and fusion head on the same frozen Temporal-Only encoders (5 seeds, tuned head config of the
    original Hybrid, same training routine and early stopping on `validation`);
  * XGBoost-Matched (tuned config, 38 columns instead of 39; deterministic, so one fit).
Test AUROC / AUPRC are compared with the stored predictions of the models that include the count; the difference (without minus
with) has a patient-cluster bootstrap 95% CI (1,000 resamples, paired draws) for every seed, and the seed mean +- SD is reported
separately. Nothing is saved to the existing prediction files or checkpoints.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
from m2_hybrid_common import DEVICE, HYBRID_STATIC_COLS  # noqa: E402
from m2_v2_common import (prepare_temporal_tensors_v2, prepare_static_features_v2, load_frozen_encoder_v2,  # noqa: E402
                          train_on_frozen_encoder_v2)
from m2_xgboost_matched_v2 import build_matched_features_v2  # noqa: E402

COMORB = HYBRID_STATIC_COLS.index("comorbidity_elixhauser_count")
KEEP = [i for i in range(len(HYBRID_STATIC_COLS)) if i != COMORB]


def main():
    fr = C.load_frames()
    fit, val, te = fr["fit"], fr["val"], fr["test"]
    cfg = json.load(open(os.path.join(C.DATA, "m2_v2_hybrid_head_tuned_config.json")))
    Xf, Mf, Df, Pf, yf = prepare_temporal_tensors_v2(fit, "train_v2", device=DEVICE)
    Xv, Mv, Dv, Pv, yv = prepare_temporal_tensors_v2(val, "train_v2", device=DEVICE)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(te, "test_v2", device=DEVICE)
    Sf, Sv, St = (torch.tensor(prepare_static_features_v2(d)[:, KEEP], device=DEVICE) for d in (fit, val, te))
    y_test = yt.cpu().numpy().astype(int)
    boot = C.ClusterBoot(te["subject_id"].values, B=1000)

    preds = pd.read_csv(os.path.join(C.DATA, "m2_v2_predictions.csv"))
    pos = pd.Series(np.arange(len(te)), index=te["stay_id"].values)

    def stored(model, seed):
        g = preds[(preds["model"] == model) & (preds["seed"] == seed)]
        out = np.empty(len(te)); out[pos.loc[g["stay_id"].values].values] = g["y_prob"].values
        return out

    res = {"dropped_feature": "comorbidity_elixhauser_count", "static_columns_kept": [HYBRID_STATIC_COLS[i] for i in KEEP],
           "bootstrap": "patient-cluster, 1,000 resamples, paired draws, 95% percentile CI", "hybrid": {}, "xgboost": {}}
    d_auc, d_prc = [], []
    for seed in C.SEEDS:
        enc = load_frozen_encoder_v2(seed)
        t0 = time.perf_counter()
        head, hist = train_on_frozen_encoder_v2(enc, Xf, Mf, Df, Pf, yf.cpu().numpy(), Xv, Mv, Dv, Pv, yv.cpu().numpy(), seed=seed,
                                                S_fit=Sf, S_val=Sv, static_hidden=cfg["static_hidden"], static_dropout=cfg["static_dropout"],
                                                fusion_dropout=cfg["fusion_dropout"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                                                max_epochs=cfg["max_epochs"], patience=cfg["patience"])
        enc.eval(); head.eval()
        with torch.no_grad():
            p_wo = torch.sigmoid(head(enc(Xt, Mt, Dt, Pt), St)).cpu().numpy().ravel()
        p_w = stored("Hybrid GRU-D v2", seed)
        s_w, s_wo = C.Scorer(y_test, p_w), C.Scorer(y_test, p_wo)
        a_w, a_wo, r_w, r_wo = s_w.auroc(), s_wo.auroc(), s_w.auprc(), s_wo.auprc()
        da = np.array([s_wo.auroc(w) - s_w.auroc(w) for w in boot.weights()])
        dp = np.array([s_wo.auprc(w) - s_w.auprc(w) for w in boot.weights()])
        res["hybrid"][seed] = {"epochs": len(hist), "seconds": time.perf_counter() - t0,
                               "auroc_with": a_w, "auroc_without": a_wo, "delta_auroc": a_wo - a_w, "delta_auroc_ci95": C.pct_ci(da),
                               "auprc_with": r_w, "auprc_without": r_wo, "delta_auprc": r_wo - r_w, "delta_auprc_ci95": C.pct_ci(dp)}
        d_auc.append(a_wo - a_w); d_prc.append(r_wo - r_w)
        print(f"seed {seed}: AUROC with {a_w:.4f} without {a_wo:.4f} delta {a_wo - a_w:+.4f} {C.pct_ci(da)}", flush=True)
    res["hybrid_seed_summary"] = {"delta_auroc": C.seed_summary(d_auc), "delta_auprc": C.seed_summary(d_prc),
                                  "auroc_without": C.seed_summary([res["hybrid"][s]["auroc_without"] for s in C.SEEDS]),
                                  "auroc_with": C.seed_summary([res["hybrid"][s]["auroc_with"] for s in C.SEEDS])}

    # XGBoost-Matched without the count (deterministic: one fit)
    tuned = json.load(open(os.path.join(C.DATA, "m2_v2_xgboost_tuned_config.json")))["effective"]
    Xfit = build_matched_features_v2(fit, "train_v2").drop(columns=["comorbidity_elixhauser_count"])
    Xte = build_matched_features_v2(te, "test_v2").drop(columns=["comorbidity_elixhauser_count"])
    m = xgb.XGBClassifier(random_state=42, **tuned).fit(Xfit, fit["outcome_critical"].astype(int).values)
    p_wo = m.predict_proba(Xte)[:, 1]
    p_w = stored("XGBoost-Matched v2", 42)
    s_w, s_wo = C.Scorer(y_test, p_w), C.Scorer(y_test, p_wo)
    da = np.array([s_wo.auroc(w) - s_w.auroc(w) for w in boot.weights()])
    dp = np.array([s_wo.auprc(w) - s_w.auprc(w) for w in boot.weights()])
    res["xgboost"] = {"n_columns": Xfit.shape[1], "auroc_with": s_w.auroc(), "auroc_without": s_wo.auroc(),
                      "delta_auroc": s_wo.auroc() - s_w.auroc(), "delta_auroc_ci95": C.pct_ci(da),
                      "auprc_with": s_w.auprc(), "auprc_without": s_wo.auprc(), "delta_auprc": s_wo.auprc() - s_w.auprc(),
                      "delta_auprc_ci95": C.pct_ci(dp)}
    print("XGBoost:", res["xgboost"]["auroc_with"], res["xgboost"]["auroc_without"], res["xgboost"]["delta_auroc_ci95"])
    C.dump(res, "phase1", "p1_3_elixhauser_ablation.json")


if __name__ == "__main__":
    main()
