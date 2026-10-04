"""
M2 5-Seed Evaluation Sweep
============================
Computes Mean +/- Std across seeds [42, 123, 7, 456, 789] for GRU-D + MAPIE
and GRU-D + ECI (lr=0.05, the newly-adopted primary configuration -- see
m2_ablations.py's ECI_LR and m2_eci_lr_sweep.py), overall and per clinical
horizon (1h/6h/12h/All).

Design choice: the Gate-mandated patient-level train/test split
(M2_Outcome_Definition_Gate.md Section 7) stays FIXED across all seeds
(random_state=42, already leak-verified -- zero patient overlap). Only the
model-training seed and the MAPIE fit/calibration-fold split vary with each
seed. This isolates model/calibration stochasticity on a single, stable
40,243-patient evaluation cohort -- the standard "same benchmark split,
multiple training runs" multi-seed protocol -- rather than re-deriving (and
re-verifying leakage on) a new cohort 5 times.

Output:
  - data/m2_seed_results/seed_{N}.json  (one per seed, raw per-seed numbers)
  - consumed by m2_multiseed_analyze.py to produce the aggregated tables.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR   = os.path.join(DATA_DIR, "m2_seed_results")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_ablations import SKLearnGRUD, prepare_data, run_mapie, run_eci, TARGET, ECI_LR, patient_level_split

SEEDS = [42, 123, 7, 456, 789]
HORIZONS = [('1h', 1.0), ('6h', 6.0), ('12h', 12.0), ('All', 9999.0)]

print("Loading data (fixed Gate-mandated patient-level split, identical for every seed)...")
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
if 'intime' in df_test.columns:
    df_test['intime'] = pd.to_datetime(df_test['intime'])
    df_test = df_test.sort_values('intime')

X_train, y_train = prepare_data(df_train)
X_test,  y_test  = prepare_data(df_test)
train_subjects = df_train["subject_id"].values

print(f"ECI_LR (primary config, adopted from m2_eci_lr_sweep.py sign-off): {ECI_LR}")

for seed in SEEDS:
    out_path = os.path.join(SEED_DIR, f"seed_{seed}.json")
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

    # Fit/calibration split varies with seed -- MAPIE's calibration fold is
    # itself a source of run-to-run variance we want the multi-seed protocol
    # to capture, same as the model-training seed. Patient-level (not
    # row-level): 34% of patients have >1 ED visit, so a row-level split
    # leaks the same patient across fit and cal -- see patient_level_split().
    X_fit, X_cal, y_fit, y_cal, _, _ = patient_level_split(
        X_train, y_train, train_subjects, test_size=0.2, seed=seed
    )

    model = SKLearnGRUD(epochs=150, lr=0.01, seed=seed)
    model.fit(X_fit, y_fit)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob_test)
    auprc = average_precision_score(y_test, y_prob_test)

    y_prob_m, cov_m, wid_m, lat_m = run_mapie(model, X_cal, y_cal, X_test, y_test)
    y_prob_e, cov_e, wid_e, lat_e, lo_e, hi_e = run_eci(model, X_test, y_test)

    print(f"  Overall: AUROC={auroc:.4f} AUPRC={auprc:.4f}  "
          f"MAPIE cov={cov_m:.4f} wid={wid_m:.4f}  ECI cov={cov_e:.4f} wid={wid_e:.4f}")

    horizon_results = {}
    for h_label, h_max in HORIZONS:
        mask = df_test['ed_los_hours'].fillna(9999) <= h_max
        df_h = df_test[mask]
        if len(df_h) < 5 or df_h[TARGET].nunique() < 2:
            horizon_results[h_label] = None
            print(f"  Horizon {h_label}: not enough data (n={len(df_h)}, skipping).")
            continue
        Xh, yh = prepare_data(df_h)
        yph = model.predict_proba(Xh)[:, 1]
        auroc_h = roc_auc_score(yh, yph)
        auprc_h = average_precision_score(yh, yph)
        _, cov_mh, wid_mh, _ = run_mapie(model, X_cal, y_cal, Xh, yh)
        if len(yh) >= 10:
            _, cov_eh, wid_eh, _, _, _ = run_eci(model, Xh, yh, scores=np.abs(yh - yph))
        else:
            cov_eh, wid_eh = float('nan'), float('nan')
        horizon_results[h_label] = {
            'n': int(len(df_h)), 'auroc': float(auroc_h), 'auprc': float(auprc_h),
            'coverage_mapie': float(cov_mh), 'width_mapie': float(wid_mh),
            'coverage_eci': float(cov_eh), 'width_eci': float(wid_eh),
        }
        print(f"  Horizon {h_label:>4}: n={len(df_h):>6}  AUROC={auroc_h:.4f}  "
              f"Cov_MAPIE={cov_mh:.4f}  Cov_ECI={cov_eh:.4f}")

    result = {
        'seed': seed,
        'eci_lr': ECI_LR,
        'overall': {
            'auroc': float(auroc), 'auprc': float(auprc),
            'coverage_mapie': float(cov_m), 'width_mapie': float(wid_m), 'latency_mapie_ms': float(lat_m),
            'coverage_eci': float(cov_e), 'width_eci': float(wid_e), 'latency_eci_ms': float(lat_e),
        },
        'horizons': horizon_results,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  -> Saved {out_path}")

print("\nAll seeds complete.")
