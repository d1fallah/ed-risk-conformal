"""M2 Track B -- serving core: raw vitals -> risk score + interval + alert.

Loads ONLY from track_b/models (ONNX, scalers, recalibration state, model card). It has no
dependency on PyTorch or on Track A code, so a service can import it directly. The FastAPI layer
will be a thin wrapper around `M2Runtime.score`.

Latency note: `latency_ms` measures preprocessing + ONNX inference + interval lookup. It excludes
HTTP/JSON overhead, which the service build adds and measures separately.
"""
import hashlib
import json
import os
import time

import numpy as np
import onnxruntime as ort

from recalibration import RecalibratingInterval

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(HERE, "models")
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"]
STATIC = ["age", "sex", "triage_acuity", "comorbidity_elixhauser_count"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class M2Runtime:
    def __init__(self, models_dir=DEFAULT_MODELS_DIR):
        self.dir = models_dir
        with open(os.path.join(models_dir, "model_card.json")) as f:
            self.card = json.load(f)
        onnx_path = os.path.join(models_dir, self.card["artifact"])
        if _sha256(onnx_path) != self.card["artifact_sha256"]:
            raise RuntimeError("ONNX artifact does not match its model card hash -- refusing to serve")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        with open(os.path.join(models_dir, "m2_scalers.json")) as f:
            sc = json.load(f)
        assert [c for c in sc["static"]["cols"]] == STATIC and self.card["vitals_order"] == VITALS
        self.v_mean = np.array([sc["vitals"]["mean"][c] for c in VITALS], dtype=np.float64)
        self.v_sd = np.array([sc["vitals"]["sd"][c] for c in VITALS], dtype=np.float64)
        self.s_mean = np.asarray(sc["static"]["mean"], dtype=np.float64)
        self.s_sd = np.asarray(sc["static"]["sd"], dtype=np.float64)
        self.threshold = float(self.card["alert_threshold"]["value"])
        self.horizon_h = int(self.card["horizon_h"])
        self.interval = RecalibratingInterval.load(os.path.join(models_dir, "m2_recalibration_state.json"))

    # -- preprocessing identical to Track A's prepare_temporal_tensors_v2 / prepare_static_features_v2
    def preprocess(self, values, obs_mask, delta_hours, pad_mask, static_raw):
        values = np.asarray(values, np.float64)
        obs = np.asarray(obs_mask, np.float64)
        X = np.where(obs.astype(bool), (values - self.v_mean) / self.v_sd, 0.0)
        S = np.nan_to_num((np.asarray(static_raw, np.float64) - self.s_mean) / self.s_sd, nan=0.0)
        return {"X": X.astype(np.float32), "M": obs.astype(np.float32),
                "D": np.asarray(delta_hours, np.float32), "pad": np.asarray(pad_mask, np.float32),
                "S": S.astype(np.float32)}

    def predict_batch(self, values, obs_mask, delta_hours, pad_mask, static_raw):
        feeds = self.preprocess(values, obs_mask, delta_hours, pad_mask, static_raw)
        return self.session.run(["prob"], feeds)[0].ravel()

    def score(self, values, obs_mask, delta_hours, pad_mask, static_raw):
        """One patient: arrays shaped (11,7), (11,7), (11,7), (11,), (4,)."""
        t0 = time.perf_counter()
        p = float(self.predict_batch(values[None], obs_mask[None], delta_hours[None], pad_mask[None],
                                     static_raw[None])[0])
        lo, hi = self.interval.interval(p)
        return {"risk_score": p, "confidence_interval": [lo, hi], "deterioration_horizon_h": self.horizon_h,
                "alert": bool(p >= self.threshold), "mode": "model", "track": "B",
                "data_label": self.card["data_label"], "latency_ms": (time.perf_counter() - t0) * 1000.0}

    def add_label(self, p, y):
        """Feed back a patient's outcome once known (drives periodic recalibration)."""
        self.interval.add_label(p, y)
