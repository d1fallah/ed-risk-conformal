"""M2 Track B -- tests that must pass before the model is served.

Run:   .venv/Scripts/python.exe track_b/test_model.py      (also collectable by pytest)
Writes track_b/results/export_report.json. A model is CLEARED only if every test passes.

Track A code is imported read-only, as the reference the exported model must reproduce.
Every number written to the report is Track B (labelled) and never enters the papers.
"""
import os, sys, json, time, datetime, traceback
os.environ["M2_V2_HORIZON"] = "6h"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "Benchmark_scripts"))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from m2_runtime import M2Runtime, STATIC
from recalibration import RecalibratingInterval
from news2_fallback import partial_news2
from aligned import chronological_aligned, assert_aligned

REPORT = {"module": "M2", "track": "B", "data_label": "Proxy/Simulated (MIMIC-IV-ED)", "tests": {}}
_CACHE = {}


def _data():
    """Raw (unscaled) arrays for the 6 h validation / calibration / test sets, built with Track A's cohort code."""
    if _CACHE:
        return _CACHE
    from m2_hybrid_common import add_hybrid_static_features, HYBRID_STATIC_COLS
    from m2_v2_common import (load_split_frame, load_dev_pool_split_v2, load_sequences, apply_horizon_filter, TARGET)
    df_train = add_hybrid_static_features(load_split_frame("train_v2"))
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    dev = load_dev_pool_split_v2()
    ids = {k: set(dev[k]["subject_ids"].tolist()) for k in ("validation", "final_conformal_calibration")}
    frames = {
        "val": (apply_horizon_filter(df_train[df_train["subject_id"].isin(ids["validation"])]), "train_v2"),
        "cal": (chronological_aligned(apply_horizon_filter(df_train[df_train["subject_id"].isin(ids["final_conformal_calibration"])])), "train_v2"),
        "test": (chronological_aligned(apply_horizon_filter(df_test)), "test_v2"),
    }
    out = {}
    for k, (df, split) in frames.items():
        values, obs, pad, delta, _ = load_sequences(split)
        assert_aligned(df, split)
        pos = df.index.values
        out[k] = {"values": values[pos], "obs": obs[pos], "pad": pad[pos], "delta": delta[pos],
                  "static": df[HYBRID_STATIC_COLS].values.astype(np.float64),
                  "y": df[TARGET].astype(int).values, "df": df}
    _CACHE.update(out)
    return _CACHE


def _predict(rt, d, bs=4096):
    return np.concatenate([rt.predict_batch(d["values"][i:i + bs], d["obs"][i:i + bs], d["delta"][i:i + bs],
                                            d["pad"][i:i + bs], d["static"][i:i + bs])
                           for i in range(0, len(d["y"]), bs)])


def _preds(rt):
    if "p" not in _CACHE:
        d = _data()
        _CACHE["p"] = {k: _predict(rt, d[k]) for k in ("val", "cal", "test")}
    return _CACHE["p"]


# ------------------------------------------------------------------------------------------
def test_loads_from_models_dir():
    rt = M2Runtime()
    card = rt.card
    assert card["track"] == "B" and "Proxy" in card["data_label"]
    assert [i.name for i in rt.session.get_inputs()] == ["X", "M", "D", "pad", "S"]
    REPORT["tests"]["loads_from_models_dir"] = {"pass": True, "artifact_sha256": card["artifact_sha256"]}


def test_matches_trackA_pytorch():
    """Exported ONNX (through the serving core) vs Track A's own PyTorch loaders, full test set."""
    import torch
    from m2_v2_common import (load_frozen_encoder_v2, load_fusion_head_v2, predict_proba_frozen_v2)
    from m2_hybrid_common import DEVICE
    rt = M2Runtime()
    d = _data()["test"]
    cfg = json.load(open(os.path.join(ROOT, "data", "m2_v2_hybrid_head_tuned_config.json")))
    enc = load_frozen_encoder_v2(42)
    head = load_fusion_head_v2("m2_hybrid_grud_head", 42, static_hidden=cfg["static_hidden"],
                               static_dropout=cfg["static_dropout"], fusion_dropout=cfg["fusion_dropout"])
    # build the tensors with Track A's preprocessing (not the serving core's)
    from m2_v2_common import prepare_temporal_tensors_v2, prepare_static_features_v2
    X, M, D, P, _ = prepare_temporal_tensors_v2(d["df"], "test_v2", device=DEVICE)
    S = torch.tensor(prepare_static_features_v2(d["df"]), device=DEVICE)
    ref = np.concatenate([predict_proba_frozen_v2(enc, head, X[i:i + 4096], M[i:i + 4096], D[i:i + 4096],
                                                  P[i:i + 4096], S=S[i:i + 4096]) for i in range(0, len(X), 4096)])
    got = _preds(rt)["test"]
    diff = float(np.abs(ref - got).max())
    REPORT["tests"]["matches_trackA_pytorch"] = {"pass": diff < 1e-5, "max_abs_diff": diff, "n": int(len(ref))}
    assert diff < 1e-5, f"ONNX/serving core differs from Track A PyTorch by {diff}"


def test_reproduces_trackA_auroc():
    rt = M2Runtime()
    y = _data()["test"]["y"]
    auroc = float(roc_auc_score(y, _preds(rt)["test"]))
    ref = rt.card["track_a_reference"]["auroc"]
    ok = round(auroc, 4) == round(ref, 4)
    REPORT["tests"]["reproduces_trackA_auroc"] = {"pass": ok, "auroc": auroc, "track_a_seed42_hz6h_auroc": ref}
    assert ok, f"AUROC {auroc:.4f} != Track A {ref:.4f}"


def test_recalibration_matches_trackA():
    """The serving-side class must produce exactly Track A's periodic-recalibration intervals (W=10k, R=2k)."""
    from m2_eci_vs_recalibration import recal_interval, static_interval
    rt = M2Runtime()
    d, p = _data(), _preds(rt)
    p_cal, y_cal, p_te, y_te = p["cal"], d["cal"]["y"].astype(float), p["test"], d["test"]["y"]
    sp, sy, nc = np.concatenate([p_cal, p_te]), np.concatenate([y_cal, y_te.astype(float)]), len(p_cal)
    lo_ref, hi_ref = recal_interval(sp, sy, nc, 10000, 2000)

    # serving mode: interval first, label afterwards, one arrival at a time
    ri = RecalibratingInterval(p_cal, y_cal, window=10000, refresh_every=2000)
    assert abs(ri.q - rt.interval.q) < 1e-12, "exported recalibration state does not match a fresh fit"
    lo, hi = np.zeros(len(p_te)), np.zeros(len(p_te))
    for i, (pi, yi) in enumerate(zip(p_te, y_te)):
        lo[i], hi[i] = ri.interval(pi)
        ri.add_label(pi, yi)
    match = bool(np.allclose(lo, lo_ref, atol=1e-12) and np.allclose(hi, hi_ref, atol=1e-12))

    # T7 analogue: window = whole calibration fold and no refresh reproduces the static interval
    lo_s, hi_s = static_interval(p_cal, y_cal, p_te)
    r7 = RecalibratingInterval(p_cal, y_cal, window=nc, refresh_every=len(p_te) + 1)
    lo7 = np.array([r7.interval(pi)[0] for pi in p_te]); hi7 = np.array([r7.interval(pi)[1] for pi in p_te])
    t7 = bool(np.allclose(lo7, lo_s) and np.allclose(hi7, hi_s))
    REPORT["tests"]["recalibration_matches_trackA"] = {"pass": match and t7, "stream_match": match, "static_reproduction": t7}
    assert match and t7


def test_interval_coverage_track_b_metric():
    """Track B metric: coverage of the periodic-recalibration interval on the 6 h test stream.
    Pass band: marginal coverage >= 0.89, the validity band pre-specified for this comparison in Track A.
    Coverage for the patients who deteriorate is REPORTED, not gated (Track A found it low)."""
    rt = M2Runtime()
    d, p = _data(), _preds(rt)
    ri = RecalibratingInterval.from_state(rt.interval.state())
    p_te, y_te = p["test"], d["test"]["y"]
    lo, hi = np.zeros(len(p_te)), np.zeros(len(p_te))
    for i, (pi, yi) in enumerate(zip(p_te, y_te)):
        lo[i], hi[i] = ri.interval(pi)
        ri.add_label(pi, yi)
    covered = (y_te >= lo) & (y_te <= hi)
    res = {"marginal_coverage": float(covered.mean()), "coverage_positive_class": float(covered[y_te == 1].mean()),
           "coverage_negative_class": float(covered[y_te == 0].mean()), "mean_width": float((hi - lo).mean()),
           "n_positive": int((y_te == 1).sum()), "n": int(len(y_te)), "labels_assumed_immediate": True}
    res["pass"] = res["marginal_coverage"] >= 0.89
    REPORT["tests"]["interval_coverage_track_b_metric"] = res
    assert res["pass"], f"marginal coverage {res['marginal_coverage']:.4f} < 0.89"


def test_alert_threshold_track_b_metric():
    """Threshold was chosen on validation for >=95% recall; recompute it and report test behaviour."""
    rt = M2Runtime()
    d, p = _data(), _preds(rt)
    thr = rt.threshold
    val_recall = float(((p["val"] >= thr) & (d["val"]["y"] == 1)).sum() / (d["val"]["y"] == 1).sum())
    pred, y = p["test"] >= thr, d["test"]["y"] == 1
    tp = int((pred & y).sum())
    res = {"threshold": thr, "validation_recall": val_recall, "test_recall": tp / int(y.sum()),
           "test_precision": tp / max(int(pred.sum()), 1), "test_alert_rate": float(pred.mean()),
           "test_positive_rate": float(y.mean()), "n_test": int(len(y))}
    res["pass"] = val_recall >= 0.95
    REPORT["tests"]["alert_threshold_track_b_metric"] = res
    assert res["pass"], f"validation recall {val_recall:.4f} < 0.95"


def test_news2_fallback_matches_trackA():
    from helpers import add_score_NEWS2
    d = _data()["test"]["df"].head(20000).copy()
    add_score_NEWS2(d)
    mine = np.array([partial_news2(r.triage_resprate, r.triage_temperature, r.triage_sbp, r.triage_heartrate,
                                   r.triage_o2sat)["news2_partial"] for r in d.itertuples()])
    ok = bool((mine == d["score_NEWS2"].values).all())
    REPORT["tests"]["news2_fallback_matches_trackA"] = {"pass": ok, "n": int(len(d)),
                                                        "n_mismatch": int((mine != d["score_NEWS2"].values).sum())}
    assert ok


def test_latency():
    rt = M2Runtime()
    d = _data()["test"]
    n = 1000
    args = [(d["values"][i], d["obs"][i], d["delta"][i], d["pad"][i], d["static"][i]) for i in range(n + 50)]
    for a in args[:50]:
        rt.score(*a)
    ms = np.array([rt.score(*a)["latency_ms"] for a in args[50:]])
    res = {"n": n, "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
           "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max()),
           "scope": "in-process: preprocessing + ONNX (CPU, 1 thread) + interval; excludes HTTP/JSON overhead",
           "targets": {"risk_score_endpoint_ms": 100, "update_s": 2.0}}
    res["pass"] = res["p95_ms"] < 100.0 and res["max_ms"] < 2000.0
    REPORT["tests"]["latency"] = res
    assert res["pass"], res


TESTS = [test_loads_from_models_dir, test_matches_trackA_pytorch, test_reproduces_trackA_auroc,
         test_recalibration_matches_trackA, test_interval_coverage_track_b_metric,
         test_alert_threshold_track_b_metric, test_news2_fallback_matches_trackA, test_latency]


def main():
    failures = []
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            failures.append(t.__name__)
            REPORT["tests"].setdefault(t.__name__.replace("test_", ""), {"pass": False})["error"] = repr(e)
            print(f"FAIL  {t.__name__}: {e}", flush=True)
            traceback.print_exc()
    REPORT["cleared"] = not failures
    REPORT["failed"] = failures
    REPORT["run_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "export_report.json"), "w") as f:
        json.dump(REPORT, f, indent=2)
    print("\nCLEARED" if not failures else f"\nNOT CLEARED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
