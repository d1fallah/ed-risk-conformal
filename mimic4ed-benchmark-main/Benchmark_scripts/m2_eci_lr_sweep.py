"""
M2 ECI Learning-Rate Sweep
===========================
Follow-up to Ablation 1 (m2_ablations.py), which found GRU-D + ECI's average
interval width (0.4424) is ~56% wider than GRU-D + MAPIE's (0.2831) on the
full 83,332-row MIMIC-IV-ED test set, despite both hitting ~90% coverage.

ECI_LR was previously fixed at 1.0 -- the top of the ECI paper's own
validated grid (Error-quantified-Conformal-Inference/tests/configs/*.yaml:
lr in {0.05, 0.1, 0.5, 1.0}) -- specifically because the ~41-point demo test
set needed the largest available step size to converge within so few
updates. That constraint no longer applies at full scale (83,332 streaming
test points). This script sweeps the same grid the paper itself validated,
using the *same* trained GRU-D model and *same* test stream as Ablation 1,
to check whether a smaller step size reduces ECI's average width without
sacrificing its ~90% coverage.

Output: data/m2_eci_lr_sweep.csv
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
ECI_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Error-quantified-Conformal-Inference"))
sys.path.insert(0, ECI_DIR)
from core.methods import ECI as eci_method

sys.path.insert(0, SCRIPT_DIR)
from m2_ablations import SKLearnGRUD, prepare_data, _sorted_bounds, patient_level_split, run_mapie

print("Loading data...")
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

if 'intime' in df_test.columns:
    df_test['intime'] = pd.to_datetime(df_test['intime'])
    df_test = df_test.sort_values('intime')

X_train, y_train = prepare_data(df_train)
X_test,  y_test  = prepare_data(df_test)
train_subjects = df_train["subject_id"].values

# Identical fit/calibration split (same seed) as m2_ablations.py Ablation 1,
# so the trained model -- and therefore y_prob on the test stream -- is
# bit-for-bit the same one evaluated there. Patient-level (not row-level):
# 34% of patients have >1 ED visit, so a row-level split leaks the same
# patient across fit and cal -- see patient_level_split() in m2_ablations.py.
X_fit, X_cal, y_fit, y_cal, _, _ = patient_level_split(
    X_train, y_train, train_subjects, test_size=0.2, seed=42
)

print("Training GRU-D (identical config/seed to m2_ablations.py Ablation 1)...")
model = SKLearnGRUD(epochs=150, lr=0.01)
model.fit(X_fit, y_fit)

y_prob = model.predict_proba(X_test)[:, 1]
scores = np.abs(y_test.astype(float) - y_prob)

print(f"\n{'ECI lr':>8} {'Coverage':>10} {'Width':>8} {'Latency(ms)':>12}")
print("-" * 44)

rows = []
for lr in [0.05, 0.1, 0.5, 1.0]:
    t0 = time.perf_counter()
    res = eci_method(scores=scores, alpha=0.1, lr=lr, T_burnin=5, ahead=1, proportional_lr=True)
    lat = (time.perf_counter() - t0) * 1000 / len(X_test)
    qs = res['q']
    lo, hi = _sorted_bounds(y_prob - qs, y_prob + qs)
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    cov = np.mean((y_test >= lo) & (y_test <= hi))
    wid = np.mean(hi - lo)
    print(f"{lr:>8.2f} {cov:>10.4f} {wid:>8.4f} {lat:>12.4f}")
    rows.append({'eci_lr': lr, 'coverage': round(cov, 4), 'width': round(wid, 4), 'latency_ms': round(lat, 4)})

df = pd.DataFrame(rows)
out_path = os.path.join(DATA_DIR, "m2_eci_lr_sweep.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
# The MAPIE reference is COMPUTED here, on the same model, split and test
# stream as the sweep above. It was previously a pair of hard-coded literals
# ("coverage=0.9004, width=0.3809") printed as though it were a measured
# result, and they had gone stale against the current config: the printed
# width was 0.3809 while m2_ablations.py's Ablation 1 -- which this line
# explicitly claims to be quoting -- measures 0.2294 for the same quantity.
# Hand-typed metrics are prohibited outright (CLAUDE.md Sec 1, the Golden
# Rule); a stale one presented as a cross-check is worse than none, because
# every "width vs. MAPIE" ratio derived from it is wrong by the same factor.
_, cov_mapie, wid_mapie, _ = run_mapie(model, X_cal, y_cal, X_test, y_test)
print(f"\nFor reference -- MAPIE, computed here on the same model, split and "
      f"test stream as the sweep above: coverage={cov_mapie:.4f}, width={wid_mapie:.4f}")
print("ECI width vs. MAPIE, per lr:")
for r in rows:
    print(f"  lr={r['eci_lr']:<5} width={r['width']:.4f}  "
          f"({(r['width'] / wid_mapie - 1) * 100:+.1f}% vs MAPIE)")
