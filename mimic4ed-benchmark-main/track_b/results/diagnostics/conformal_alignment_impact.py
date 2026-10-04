import os, sys, json
ROOT = r"C:\Users\user\Documents\Modules Services startup\module2\mimic4ed-benchmark-main"
sys.path.insert(0, os.path.join(ROOT, "Benchmark_scripts")); sys.path.insert(0, os.path.join(ROOT, "track_b"))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, run_eci_4way, DEVICE
from m2_v2_common import (load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2)
from m2_conformal_v2 import get_predictions_v2
from m2_eci_vs_recalibration import static_interval, recal_interval
from aligned import chronological_aligned

SEED = 42
df_train = add_hybrid_static_features(load_split_frame("train_v2"))
df_test = add_hybrid_static_features(load_split_frame("test_v2"))
dev = load_dev_pool_split_v2()
cs = set(dev["calibration_selection"]["subject_ids"].tolist()); fc = set(dev["final_conformal_calibration"]["subject_ids"].tolist())

def run(sorter, label):
    df_cs = sorter(df_train[df_train["subject_id"].isin(cs)]); df_fc = sorter(df_train[df_train["subject_id"].isin(fc)]); df_te = sorter(df_test)
    out = {}
    for name, df, split in (("cs", df_cs, "train_v2"), ("fc", df_fc, "train_v2"), ("te", df_te, "test_v2")):
        X, M, D, P, y = prepare_temporal_tensors_v2(df, split, device=DEVICE)
        S = torch.tensor(prepare_static_features_v2(df), device=DEVICE)
        out[name] = (get_predictions_v2("Hybrid GRU-D v2", SEED, X, M, D, P, S=S), y.cpu().numpy().astype(int))
    (p_cs, y_cs), (p_fc, y_fc), (p_te, y_te) = out["cs"], out["fc"], out["te"]
    res = {"test_auroc": float(roc_auc_score(y_te, p_te)), "n_test": int(len(y_te)), "n_pos": int(y_te.sum())}
    def cov(lo, hi):
        c = (y_te >= lo) & (y_te <= hi)
        return {"marginal": round(float(c.mean()), 4), "positive_class": round(float(c[y_te == 1].mean()), 4), "mean_width": round(float((hi - lo).mean()), 4)}
    res["static"] = cov(*static_interval(p_fc, y_fc.astype(float), p_te))
    sp = np.concatenate([p_fc, p_te]); sy = np.concatenate([y_fc, y_te]).astype(float)
    res["recal_W10k"] = cov(*recal_interval(sp, sy, len(p_fc), 10000, 2000))
    _lr, _c, _w, lo_e, hi_e, _cands = run_eci_4way(p_cs, y_cs, p_fc, y_fc, p_te, y_te)
    res["eci"] = cov(lo_e, hi_e)
    print(label, json.dumps(res), flush=True)
    return res

run(sort_chronologically, "AS-RUN-BY-TRACK-A (misaligned)")
run(chronological_aligned, "ALIGNED                        ")
