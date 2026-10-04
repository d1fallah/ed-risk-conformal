import os, sys
ROOT = r"C:\Users\user\Documents\Modules Services startup\module2\mimic4ed-benchmark-main"
sys.path.insert(0, os.path.join(ROOT, "Benchmark_scripts"))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, DEVICE
from m2_v2_common import (load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2)
from m2_conformal_v2 import get_predictions_v2

df_train = add_hybrid_static_features(load_split_frame("train_v2"))
df_test = add_hybrid_static_features(load_split_frame("test_v2"))
dev = load_dev_pool_split_v2()
fc = set(dev["final_conformal_calibration"]["subject_ids"].tolist())

def auroc_for(df, split, label):
    X, M, D, P, y = prepare_temporal_tensors_v2(df, split, device=DEVICE)
    S = torch.tensor(prepare_static_features_v2(df), device=DEVICE)
    p = get_predictions_v2("Hybrid GRU-D v2", 42, X, M, D, P, S=S)
    print(f"{label:55s} n={len(df):6d}  AUROC={roc_auc_score(y.cpu().numpy(), p):.4f}")

auroc_for(df_test, "test_v2", "TEST   unsorted (original order)")
auroc_for(sort_chronologically(df_test), "test_v2", "TEST   sort_chronologically (as Track A conformal)")
fold = df_train[df_train["subject_id"].isin(fc)]
auroc_for(fold, "train_v2", "CAL    unsorted boolean-filtered view")
auroc_for(sort_chronologically(fold), "train_v2", "CAL    sort_chronologically (as Track A conformal)")
