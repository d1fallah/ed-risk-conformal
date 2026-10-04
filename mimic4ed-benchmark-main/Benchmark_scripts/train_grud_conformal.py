"""
M2 Track B -- canonical GRU-D fit + conformal intervals + ONNX export
=====================================================================
This script produces the Track B deployment artifact (`grud_risk.onnx`) and
its latency measurement.

REFACTORED 2026-09-01 under M2_Hybrid_Deviation_Log.md Deviation 9
------------------------------------------------------------------
This file previously carried its OWN copy of the GRU-D recurrence and its
OWN `prepare_data`, duplicating -- and then silently diverging from -- the
implementations used everywhere else in M2. Deviation 7's repair reached
`m2_hybrid_common.py` (the v3.8 pipeline) and `m2_ablations.py` (Track A)
but never reached this file, because it duplicated rather than imported.
Verified by direct execution on the frozen test cohort before the fix:

    mask fraction observed          1.0000   (mask carried no information)
    X min / max / mean         0.00 / 269.00 / 63.39  (unscaled inputs)
    grad W_gamma absmax             0.0      (decay never trained)
    grad b_gamma absmax             0.0
    |out(D) - out(1000*D)|          0.0      (Delta-t had NO effect at all)

The exported artifact was therefore a masked GRU with zero-imputation in
which elapsed time was ignored -- not GRU-D, and not the model reported in
the dossier's Track A sections.

The fix is to delete the duplicates rather than repair them. Three
hand-copied versions of one recurrence existed in this repository; fixing
two left the third broken and undetected. This file now imports the single
shared implementation:

  * `GRUD` / `SKLearnGRUD` from `m2_ablations`, whose recurrence is the
    shared `GRUDCell` in `m2_grud_cell.py` (Che et al. 2018, Eqs. 10-13).
  * `prepare_data` from `m2_ablations`: cleaned vitals from the master
    lookup (real pre-imputation missingness, Celsius conversion, outlier
    removal), train-only standardisation of vitals and statics, and Delta-t
    in units of the training-set median ED length of stay.
  * `patient_level_split`, `unpack`, `run_mapie`, `run_eci` likewise.

TRACK SEPARATION (CLAUDE.md Sec 0). Sharing the architecture and the
preprocessing is not a Track A/Track B parameter mix: a working decay
mechanism and correctly scaled inputs are correctness, not a research
constraint, and "the decay mechanism never runs" is a defect rather than a
deployment shortcut. What stays Track-B-specific lives here and nowhere
else: the ONNX export, the CPU-resident deployment target, and the latency
benchmark. Track B remains free to diverge on serving-time policy (interval
source, caching, batching); it is not free to ship a different model than
the one that was evaluated.

Output: `grud_risk.onnx` + a measured ONNX Runtime latency.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import onnxruntime as ort

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
sys.path.insert(0, SCRIPT_DIR)

# The single shared implementation -- see the module docstring for why none
# of this is defined locally any more.
from m2_ablations import (
    TARGET, STATIC_SIZE, prepare_data, patient_level_split, unpack,
    SKLearnGRUD, run_mapie_bounds, run_eci,
)
from m2_hybrid_common import sort_chronologically


def main():
    print("Loading data...", flush=True)
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    X_train, y_train = prepare_data(df_train)
    X_test, y_test = prepare_data(df_test)
    train_subjects = df_train["subject_id"].values

    # Split-conformal's guarantee assumes the calibration set is exchangeable
    # with, and disjoint from, the set the model was fit on. The split is
    # patient-level, not row-level: 34% of patients have more than one ED
    # visit, so a row-level split leaks the same patient across fit and
    # calibration (10,156 overlapping patients at seed=42, measured directly
    # before this was fixed).
    X_fit, X_cal, y_fit, y_cal, _, _ = patient_level_split(
        X_train, y_train, train_subjects, test_size=0.2, seed=42)

    print("Training GRU-D at the adopted Track A config...", flush=True)
    # No config overrides: SKLearnGRUD's defaults ARE the adopted config
    # (hid=64, no pos_weight, no dropout/weight_decay, full batch), so the
    # deployed artifact is the same model the dossier reports rather than a
    # separately-tuned one. See M2_Findings_Dossier.md Sec 3.1.
    base_model = SKLearnGRUD(epochs=150, lr=0.01, seed=42)
    base_model.fit(X_fit, y_fit)

    y_prob = base_model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    _, coverage, avg_width, mapie_lat, _, _ = run_mapie_bounds(
        base_model, X_cal, y_cal, X_test, y_test)

    print("-" * 56)
    print("Task 2 - Critical Outcome Prediction (GRU-D + MAPIE)")
    print("-" * 56)
    print(f"AUROC:                       {auroc:.4f}")
    print(f"AUPRC:                       {auprc:.4f}")
    print(f"Empirical 90% Coverage Rate: {coverage:.4f}")
    print(f"Average Interval Width:      {avg_width:.4f}")

    # --- ECI --------------------------------------------------------------
    # ECI is an online method, so the order of the test stream is part of the
    # method. This previously sorted by raw `intime`, which is wrong for
    # MIMIC-IV: dates are shifted by a random per-patient offset (anchor_year
    # spans 2110-2208 in this cohort), so `intime` is not comparable across
    # patients and sorting on it yields an arbitrary order that merely looks
    # chronological. sort_chronologically() de-shifts via anchor_year_group;
    # see m2_hybrid_common._load_real_time_lookup() for the recipe and its
    # +/-1 year resolution limit.
    print("-" * 56)
    print("Task 2 - Critical Outcome Prediction (GRU-D + ECI)")
    print("-" * 56)
    df_test_sorted = sort_chronologically(df_test)
    X_test_sorted, y_test_sorted = prepare_data(df_test_sorted)

    # lr=0.05 is the adopted primary configuration, re-confirmed as the
    # narrowest point on the ECI authors' own validated grid after the
    # config change -- see data/m2_eci_lr_sweep.csv and dossier Sec 5.
    _, eci_coverage, eci_avg_width, eci_lat, _, _ = run_eci(
        base_model, X_test_sorted, y_test_sorted)
    print(f"Empirical 90% Coverage Rate: {eci_coverage:.4f}")
    print(f"Average Interval Width:      {eci_avg_width:.4f}")

    # --- ONNX export ------------------------------------------------------
    print("-" * 56)
    print("Exporting GRU-D to ONNX format ('grud_risk.onnx')...", flush=True)
    # Exported from CPU regardless of training device: the artifact is a
    # deployment target (Track B, ONNX Runtime) and must be portable.
    base_model.model.to("cpu").eval()
    dummy_X, dummy_M, dummy_Delta, dummy_S = unpack(X_test[:1])
    onnx_path = os.path.join(SCRIPT_DIR, "grud_risk.onnx")
    torch.onnx.export(
        base_model.model,
        (dummy_X.cpu(), dummy_M.cpu(), dummy_Delta.cpu(), dummy_S.cpu()),
        onnx_path,
        input_names=["X", "M", "Delta", "S"],
        output_names=["output"],
        dynamic_axes={k: {0: "batch_size"} for k in ("X", "M", "Delta", "S")},
    )

    # Verify the exported graph reproduces the PyTorch model rather than
    # assuming the trace succeeded. A silent mismatch here would ship a
    # different model than the one just evaluated -- exactly the class of
    # failure Deviation 9 was.
    sess = ort.InferenceSession(onnx_path)
    Xt, Mt, Dt, St = unpack(X_test[:512])
    with torch.no_grad():
        torch_out = base_model.model(Xt.cpu(), Mt.cpu(), Dt.cpu(), St.cpu()).numpy()
    onnx_out = sess.run(None, {"X": Xt.cpu().numpy(), "M": Mt.cpu().numpy(),
                               "Delta": Dt.cpu().numpy(), "S": St.cpu().numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    assert max_diff < 1e-4, f"ONNX export does not match PyTorch (max diff {max_diff:.2e}) -- STOP."
    print(f"  ONNX vs. PyTorch agreement: max abs diff {max_diff:.2e} (OK)")

    # Delta-t must measurably affect the output. Before Deviation 9's fix
    # this was exactly 0.0 -- the check that would have caught it.
    with torch.no_grad():
        a = base_model.model(Xt.cpu(), Mt.cpu(), Dt.cpu(), St.cpu())
        b = base_model.model(Xt.cpu(), Mt.cpu(), Dt.cpu() * 5.0, St.cpu())
    delta_effect = float((a - b).abs().max())
    assert delta_effect > 0.0, "Delta-t has no effect on the exported model -- decay is inert. STOP."
    print(f"  Delta-t sensitivity |out(D)-out(5D)|: {delta_effect:.4f} (non-zero, OK)")
    print(f"  Artifact size: {os.path.getsize(onnx_path)/1024:.1f} KB")

    print("Benchmarking ONNX Runtime inference latency...", flush=True)
    X_t, M_t, Delta_t, S_t = unpack(X_test)
    X_t, M_t, Delta_t, S_t = (X_t.cpu().numpy(), M_t.cpu().numpy(),
                              Delta_t.cpu().numpy(), S_t.cpu().numpy())
    n_bench = min(5000, len(X_test))
    latencies = []
    for i in range(n_bench):
        inputs = {"X": X_t[i:i+1], "M": M_t[i:i+1],
                  "Delta": Delta_t[i:i+1], "S": S_t[i:i+1]}
        start = time.perf_counter()
        sess.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    print(f"Average Inference Latency:   {np.mean(latencies):.4f} ms/patient "
          f"(n={n_bench}, single-patient calls)")

    out = {
        "auroc": round(float(auroc), 4), "auprc": round(float(auprc), 4),
        "coverage_mapie": round(float(coverage), 4), "width_mapie": round(float(avg_width), 4),
        "coverage_eci": round(float(eci_coverage), 4), "width_eci": round(float(eci_avg_width), 4),
        "onnx_kb": round(os.path.getsize(onnx_path) / 1024, 1),
        "onnx_vs_torch_max_abs_diff": max_diff,
        "delta_t_sensitivity": round(delta_effect, 6),
        "latency_ms_per_patient": round(float(np.mean(latencies)), 4),
        "n_latency_samples": int(n_bench),
    }
    import json
    with open(os.path.join(DATA_DIR, "m2_trackB_artifact.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {os.path.join(DATA_DIR, 'm2_trackB_artifact.json')}")


if __name__ == "__main__":
    main()
