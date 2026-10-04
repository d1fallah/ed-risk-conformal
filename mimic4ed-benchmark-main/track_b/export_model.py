"""M2 Track B -- export the Hybrid GRU-D (6 h horizon, seed 42) as one ONNX model + serving bundle.

Run from module2/mimic4ed-benchmark-main with the module's .venv:
    .venv/Scripts/python.exe track_b/export_model.py

Track A code and checkpoints are imported/read only. Nothing here writes outside track_b/.
Seed 42 is fixed in advance (first locked seed); it is never chosen by test score.
Thresholds are chosen on the validation split only.
"""
import os, sys, json, hashlib, datetime
os.environ["M2_V2_HORIZON"] = "6h"           # must be set before m2_v2_common is imported

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "Benchmark_scripts"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.metrics import roc_auc_score

from m2_hybrid_common import add_hybrid_static_features
from m2_grud_cell import run_grud_masked
from m2_v2_common import (
    load_split_frame, load_dev_pool_split_v2, prepare_temporal_tensors_v2, prepare_static_features_v2,
    load_frozen_encoder_v2, load_fusion_head_v2, apply_horizon_filter, CHECKPOINT_DIR, SUFFIX,
    SCALER_PATH_V2, STATIC_SCALER_PATH_V2,
)
from recalibration import RecalibratingInterval
from aligned import chronological_aligned, assert_aligned

SEED = 42
RECALL_TARGET = 0.95
MODELS_DIR = os.path.join(HERE, "models")
CFG = json.load(open(os.path.join(ROOT, "data", "m2_v2_hybrid_head_tuned_config.json")))
assert SUFFIX == "_hz6h"
NAMES = ["X", "M", "D", "pad", "S"]


class M2RiskNet(nn.Module):
    """Frozen GRU-D encoder + static branch + fusion head as one graph: (X,M,D,pad,S) -> P(event)."""

    def __init__(self, encoder, head):
        super().__init__()
        self.cell, self.head = encoder.cell, head

    def forward(self, X, M, D, pad, S):
        h = run_grud_masked(self.cell, X, M, D, pad)
        return torch.sigmoid(self.head(h, S))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensors(df, split):
    """Returns ([X, M, D, pad, S] as float32 tensors, y as int array)."""
    X, M, D, P, y = prepare_temporal_tensors_v2(df, split, device="cpu")
    S = torch.tensor(prepare_static_features_v2(df), dtype=torch.float32)
    return [X, M, D, P, S], y.numpy().astype(int)


def threshold_for_recall(p_val, y_val, target):
    """Largest threshold whose validation recall is >= target."""
    order = np.argsort(-p_val)
    tp = np.cumsum(y_val[order])
    k = int(np.searchsorted(tp, target * y_val.sum(), side="left"))
    return float(p_val[order][k])


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    encoder = load_frozen_encoder_v2(SEED).to("cpu")
    head = load_fusion_head_v2("m2_hybrid_grud_head", SEED, static_hidden=CFG["static_hidden"],
                               static_dropout=CFG["static_dropout"], fusion_dropout=CFG["fusion_dropout"]).to("cpu")
    net = M2RiskNet(encoder, head).eval()

    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev = load_dev_pool_split_v2()
    ids = {k: set(dev[k]["subject_ids"].tolist()) for k in ("validation", "final_conformal_calibration")}
    df_val = apply_horizon_filter(df_train[df_train["subject_id"].isin(ids["validation"])])
    df_cal = chronological_aligned(apply_horizon_filter(df_train[df_train["subject_id"].isin(ids["final_conformal_calibration"])]))
    df_te = apply_horizon_filter(df_test)
    for _df, _split in ((df_val, "train_v2"), (df_cal, "train_v2"), (df_te, "test_v2")):
        assert_aligned(_df, _split)
    print(f"6h horizon rows: val={len(df_val)} calibration={len(df_cal)} test={len(df_te)}", flush=True)

    (val, y_val), (cal, y_cal), (te, y_te) = tensors(df_val, "train_v2"), tensors(df_cal, "train_v2"), tensors(df_te, "test_v2")

    # ---- ONNX export (fixed T=11 -> the recurrence unrolls; batch size is dynamic) ----------
    onnx_path = os.path.join(MODELS_DIR, "m2_hybrid_grud_hz6h_seed42.onnx")
    example = tuple(t[:2] for t in te)
    torch.onnx.export(net, example, onnx_path, input_names=NAMES, output_names=["prob"], opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in NAMES + ["prob"]}, dynamo=False)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def run_ort(t, bs=4096):
        out = []
        for i in range(0, len(t[0]), bs):
            out.append(sess.run(["prob"], {n: t[j][i:i + bs].numpy() for j, n in enumerate(NAMES)})[0].ravel())
        return np.concatenate(out)

    def run_torch(t, bs=4096):
        with torch.no_grad():
            return np.concatenate([net(*[t[j][i:i + bs] for j in range(5)]).numpy().ravel() for i in range(0, len(t[0]), bs)])

    p_val, p_cal, p_te = run_ort(val), run_ort(cal), run_ort(te)
    max_diff = max(float(np.abs(run_ort(t) - run_torch(t)).max()) for t in (val, cal, te))
    auroc = float(roc_auc_score(y_te, p_te))
    print(f"ONNX vs PyTorch max |diff| = {max_diff:.2e}; test AUROC = {auroc:.4f}", flush=True)

    # ---- validation-chosen alert threshold (>=95% recall) -----------------------------------
    thr = threshold_for_recall(p_val, y_val, RECALL_TARGET)

    # ---- recalibration state: last W labelled arrivals of the calibration fold --------------
    RecalibratingInterval(p_cal, y_cal).save(os.path.join(MODELS_DIR, "m2_recalibration_state.json"))

    # ---- bundle scalers + model card --------------------------------------------------------
    scalers = {"vitals": json.load(open(SCALER_PATH_V2)), "static": json.load(open(STATIC_SCALER_PATH_V2))}
    json.dump(scalers, open(os.path.join(MODELS_DIR, "m2_scalers.json"), "w"), indent=2)

    src = {n: os.path.join(CHECKPOINT_DIR, f"{n}_v2{SUFFIX}_seed{SEED}.pt")
           for n in ("m2_temporal_only_grud", "m2_hybrid_grud_head")}
    track_a = json.load(open(os.path.join(ROOT, "data", "m2_v2_seed_results_hz6h", "hybrid_grud", f"seed_{SEED}.json")))
    card = {
        "name": "m2_hybrid_grud_hz6h_seed42", "module": "M2", "track": "B",
        "data_label": "Proxy/Simulated (MIMIC-IV-ED)",
        "artifact": "m2_hybrid_grud_hz6h_seed42.onnx", "artifact_sha256": sha256(onnx_path),
        "source_checkpoints": {k: {"path": os.path.relpath(v, ROOT), "sha256": sha256(v)} for k, v in src.items()},
        "seed": SEED, "seed_policy": "fixed in advance (first locked seed), not chosen by test score",
        "horizon_h": 6,
        "input_shapes": {"X": "(B,11,7)", "M": "(B,11,7)", "D": "(B,11,7)", "pad": "(B,11)", "S": "(B,4)"},
        "vitals_order": ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"],
        "static_order": ["age", "sex", "triage_acuity", "comorbidity_elixhauser_count"],
        "track_a_reference": {"auroc": track_a["auroc"], "auprc": track_a["auprc"],
                              "source": "data/m2_v2_seed_results_hz6h/hybrid_grud/seed_42.json"},
        "alert_threshold": {"value": thr, "rule": f"largest validation threshold with recall >= {RECALL_TARGET}",
                            "chosen_on": "validation split (6h, horizon-eligible)"},
        "interval": {"method": "periodic recalibration, absolute residual", "alpha": 0.1, "window": 10000,
                     "refresh_every": 2000, "state_file": "m2_recalibration_state.json",
                     "calibration_fold": "final_conformal_calibration (6h, chronological)"},
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "export_checks": {"onnx_vs_torch_max_abs_diff": max_diff, "test_auroc": auroc},
    }
    json.dump(card, open(os.path.join(MODELS_DIR, "model_card.json"), "w"), indent=2)
    print("exported ->", MODELS_DIR, "| alert threshold =", round(thr, 5), flush=True)


if __name__ == "__main__":
    main()
