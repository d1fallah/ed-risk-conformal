"""
One-off fix: data/m2_hybrid_predictions.csv was originally saved with only
subject_id, which repeats for the ~34% of patients with more than one ED
visit -- insufficient to uniquely identify an encounter within a (model,
seed) block, and required for the Guide's predictions_all_models.csv
integrity checks ("unique (subject_id, encounter_id) pairs") and for
correctly constructing the patient-cluster bootstrap (Phase M). Adds
test.csv's unique stay_id to every row.

For the 3 GRU-D-family models (already-trained checkpoints exist),
predictions are RECOMPUTED from the exact saved checkpoint -- no
retraining, zero risk of any run-to-run drift changing an already-reported
number. Static-Only MLP has no saved checkpoint (its script doesn't
persist one); it is retrained at its already-known frozen config (hidden=
64, dropout=0.1, skipping the tuning phase this time) and the new numbers
are compared against the already-reported ones before being trusted.
XGBoost-Matched is retrained fresh (fast, and already confirmed
deterministic across seeds).
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    TARGET, HYBRID_STATIC_COLS, add_hybrid_static_features, build_fixed_dev_split,
    prepare_temporal_tensors, load_temporal_only_model, load_frozen_encoder, load_fusion_head,
    predict_proba_temporal_only, predict_proba_frozen,
    expected_calibration_error, calibration_slope_intercept, save_predictions,
)
from m2_static_only_mlp import StaticMLP, train_one as train_static_mlp, predict_proba as predict_static_mlp
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from m2_xgboost_matched import build_matched_features

FINAL_SEEDS = [42, 123, 7, 456, 789]

df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
df_test_static = add_hybrid_static_features(df_test)
y_test = df_test[TARGET].astype(int).values
subj_test = df_test["subject_id"].values
stay_test = df_test["stay_id"].values
X_test, M_test, D_test, _ = prepare_temporal_tensors(df_test)
S_test = df_test_static[HYBRID_STATIC_COLS].values.astype(np.float32)

# ── 1-3: GRU-D-family models -- reload checkpoint, recompute, resave ───────
seed_dirs = {
    "Temporal-Only GRU-D": "temporal_only_grud",
    "Frozen-Temporal Head Control": "frozen_temporal_control",
    "Hybrid GRU-D": "hybrid_grud",
}
for model_name, dirname in seed_dirs.items():
    print(f"\n{model_name}:")
    seed_dir = os.path.join(DATA_DIR, "m2_hybrid_seed_results", dirname)
    for seed in FINAL_SEEDS:
        with open(os.path.join(seed_dir, f"seed_{seed}.json")) as f:
            cfg = json.load(f)["config"]
        if model_name == "Temporal-Only GRU-D":
            model = load_temporal_only_model(seed)
            y_prob = predict_proba_temporal_only(model, X_test, M_test, D_test)
        elif model_name == "Frozen-Temporal Head Control":
            encoder = load_frozen_encoder(seed)
            head = load_fusion_head("m2_frozen_temporal_control_head", seed,
                                     static_hidden=None, fusion_dropout=cfg["fusion_dropout"])
            y_prob = predict_proba_frozen(encoder, head, X_test, M_test, D_test)
        else:
            encoder = load_frozen_encoder(seed)
            head = load_fusion_head("m2_hybrid_grud_head", seed, static_hidden=cfg["static_hidden"],
                                     static_dropout=cfg["static_dropout"], fusion_dropout=cfg["fusion_dropout"])
            y_prob = predict_proba_frozen(encoder, head, X_test, M_test, D_test, S=S_test)

        auroc = roc_auc_score(y_test, y_prob)
        print(f"  seed={seed}: AUROC={auroc:.4f} (recomputed from saved checkpoint, no retraining)")
        save_predictions(subj_test, model_name, seed, y_test, y_prob, stay_id=stay_test)

# ── 4: XGBoost-Matched -- fast, deterministic, retrain fresh ───────────────
print("\nXGBoost-Matched:")
dev_split = build_fixed_dev_split()
train_ids = set(dev_split["training"]["subject_ids"].tolist())
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_train_fit = df_train[df_train["subject_id"].isin(train_ids)]
X_train = build_matched_features(df_train_fit)
X_test_m = build_matched_features(df_test)
y_train = df_train_fit[TARGET].astype(int).values
for seed in FINAL_SEEDS:
    clf = xgb.XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss", verbosity=0)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test_m)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    print(f"  seed={seed}: AUROC={auroc:.4f} (expect 0.9056, matching the already-reported value)")
    save_predictions(subj_test, "XGBoost-Matched", seed, y_test, y_prob, stay_id=stay_test)

# ── 5: Static-Only MLP -- retrain at the already-known frozen config ──────
print("\nStatic-Only MLP:")
dev_split = build_fixed_dev_split()
train_ids = set(dev_split["training"]["subject_ids"].tolist())
val_ids = set(dev_split["validation"]["subject_ids"].tolist())
df_fit = df_train[df_train["subject_id"].isin(train_ids)]
df_fit = add_hybrid_static_features(df_fit)
df_val = df_train[df_train["subject_id"].isin(val_ids)]
df_val = add_hybrid_static_features(df_val)
X_fit_df = df_fit[HYBRID_STATIC_COLS].values
X_val_df = df_val[HYBRID_STATIC_COLS].values
y_fit = df_fit[TARGET].astype(int).values
y_val = df_val[TARGET].astype(int).values
HIDDEN, DROPOUT = 64, 0.1  # already-selected frozen config, see Dossier Sec 12
for seed in FINAL_SEEDS:
    scaler = StandardScaler().fit(X_fit_df)
    X_fit_s, X_val_s = scaler.transform(X_fit_df), scaler.transform(X_val_df)
    X_test_s = scaler.transform(S_test)  # HYBRID_STATIC_COLS order matches
    model = train_static_mlp(X_fit_s, y_fit, X_val_s, y_val, HIDDEN, DROPOUT, seed)
    y_prob = predict_static_mlp(model, X_test_s)
    auroc = roc_auc_score(y_test, y_prob)
    print(f"  seed={seed}: AUROC={auroc:.4f} (expect ~0.8488, matching the already-reported value)")
    save_predictions(subj_test, "Static-Only MLP", seed, y_test, y_prob, stay_id=stay_test)

print("\nDone. Verify AUROCs above match the already-reported Dossier values before trusting"
      " the resaved predictions file.")
