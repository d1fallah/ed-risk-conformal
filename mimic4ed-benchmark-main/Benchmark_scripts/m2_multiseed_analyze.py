"""
M2 5-Seed Aggregation
=======================
Reads data/m2_seed_results/seed_{N}.json (produced by m2_multiseed_run.py for
seeds [42, 123, 7, 456, 789]) and computes Mean +/- Std tables, overall and
per clinical horizon (1h/6h/12h/All), for GRU-D + MAPIE and GRU-D + ECI
(lr=0.05).

Output:
  - data/m2_multiseed_overall.csv
  - data/m2_multiseed_horizon.csv
"""
import os, json, glob
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR   = os.path.join(DATA_DIR, "m2_seed_results")

files = sorted(glob.glob(os.path.join(SEED_DIR, "seed_*.json")))
assert len(files) == 5, f"Expected 5 seed result files, found {len(files)}: {files}"

results = [json.load(open(f)) for f in files]
seeds = [r['seed'] for r in results]
print(f"Aggregating seeds: {seeds}")

def mstd(vals):
    return float(np.mean(vals)), float(np.std(vals, ddof=1))

# ── Overall (MAPIE vs ECI) ───────────────────────────────────────────────
overall_rows = []
for method, cov_key, wid_key, lat_key in [
    ('GRU-D + MAPIE', 'coverage_mapie', 'width_mapie', 'latency_mapie_ms'),
    ('GRU-D + ECI (lr=0.05)', 'coverage_eci', 'width_eci', 'latency_eci_ms'),
]:
    auroc_m, auroc_s = mstd([r['overall']['auroc'] for r in results])
    auprc_m, auprc_s = mstd([r['overall']['auprc'] for r in results])
    cov_m, cov_s = mstd([r['overall'][cov_key] for r in results])
    wid_m, wid_s = mstd([r['overall'][wid_key] for r in results])
    lat_m, lat_s = mstd([r['overall'][lat_key] for r in results])
    overall_rows.append({
        'variant': method, 'n_seeds': len(results),
        'auroc_mean': round(auroc_m, 4), 'auroc_std': round(auroc_s, 4),
        'auprc_mean': round(auprc_m, 4), 'auprc_std': round(auprc_s, 4),
        'coverage_mean': round(cov_m, 4), 'coverage_std': round(cov_s, 4),
        'width_mean': round(wid_m, 4), 'width_std': round(wid_s, 4),
        'latency_ms_mean': round(lat_m, 4), 'latency_ms_std': round(lat_s, 4),
    })
df_overall = pd.DataFrame(overall_rows)
overall_path = os.path.join(DATA_DIR, "m2_multiseed_overall.csv")
df_overall.to_csv(overall_path, index=False)
print("\n=== Overall (5-seed mean +/- std) ===")
print(df_overall.to_string(index=False))
print(f"Saved: {overall_path}")

# ── Per horizon ───────────────────────────────────────────────────────────
horizon_rows = []
for h_label in ['1h', '6h', '12h', 'All']:
    h_data = [r['horizons'].get(h_label) for r in results]
    if any(h is None for h in h_data):
        print(f"Horizon {h_label}: skipped (missing in at least one seed).")
        continue
    n_vals = [h['n'] for h in h_data]
    auroc_m, auroc_s = mstd([h['auroc'] for h in h_data])
    auprc_m, auprc_s = mstd([h['auprc'] for h in h_data])
    cov_mapie_m, cov_mapie_s = mstd([h['coverage_mapie'] for h in h_data])
    wid_mapie_m, wid_mapie_s = mstd([h['width_mapie'] for h in h_data])
    cov_eci_m, cov_eci_s = mstd([h['coverage_eci'] for h in h_data])
    wid_eci_m, wid_eci_s = mstd([h['width_eci'] for h in h_data])
    horizon_rows.append({
        'horizon': h_label, 'n': n_vals[0], 'n_seeds': len(h_data),
        'auroc_mean': round(auroc_m, 4), 'auroc_std': round(auroc_s, 4),
        'auprc_mean': round(auprc_m, 4), 'auprc_std': round(auprc_s, 4),
        'coverage_mapie_mean': round(cov_mapie_m, 4), 'coverage_mapie_std': round(cov_mapie_s, 4),
        'width_mapie_mean': round(wid_mapie_m, 4), 'width_mapie_std': round(wid_mapie_s, 4),
        'coverage_eci_mean': round(cov_eci_m, 4), 'coverage_eci_std': round(cov_eci_s, 4),
        'width_eci_mean': round(wid_eci_m, 4), 'width_eci_std': round(wid_eci_s, 4),
    })
df_horizon = pd.DataFrame(horizon_rows)
horizon_path = os.path.join(DATA_DIR, "m2_multiseed_horizon.csv")
df_horizon.to_csv(horizon_path, index=False)
print("\n=== Per-horizon (5-seed mean +/- std) ===")
print(df_horizon.to_string(index=False))
print(f"Saved: {horizon_path}")
