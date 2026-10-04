"""
M2 -- Streaming-budget latency benchmark on the v2 pipeline (CLAUDE.md M2: streaming budget
< 2.0 s per patient update, training and explanation excluded). One standardized benchmark per
update path; hardware and methodology are recorded in the output JSON.

Methodology (fixed here, not tuned):
  * batch size 1 (one patient update at a time), torch.no_grad, model in eval mode;
  * Hybrid GRU-D v2, seed 42 (frozen Temporal-Only encoder + tuned fusion head), the primary model;
  * inputs: 2,000 test patients drawn with a fixed seed from the chronologically sorted test frame,
    real v2 tensors (irregular sequence up to T_max=10 observations, masks, time deltas, statics);
  * 200 untimed warm-up calls, then every call timed individually with time.perf_counter
    (torch.cuda.synchronize() before and after on GPU); median, p95, p99 and max reported;
  * forward pass measured on GPU (cuda) and on CPU (default torch threads) separately;
  * conformal update paths measured over the full 84,241-arrival test stream and amortized:
      - MAPIE-equivalent split-conformal: threshold compare per arrival (threshold precomputed);
      - ECI: one recursion step + set decision per arrival (class ECIStep, verified equal to
        the vendored ECI radius trajectory before timing);
      - Mondrian-ECI: two per-class radii, same step, per arrival;
      - periodic recalibration: cost of ONE refresh (np.quantile over the window) at W = 10k / 40k,
        amortized over R = 2,000 arrivals;
  * end-to-end path = forward (GPU) + ECI step + set decision for each of the 2,000 patients.
Latency is hardware- and load-dependent; the figures describe this machine only.
"""
import os, sys, json, time, math, platform, copy
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import m2_label_lag as ll
from m2_hybrid_common import add_hybrid_static_features, sort_chronologically, DEVICE
from m2_v2_common import (load_split_frame, prepare_temporal_tensors_v2, prepare_static_features_v2,
                          load_frozen_encoder_v2, load_fusion_head_v2)
from m2_conformal_v2 import _tuned

OUT_JSON = os.path.join(ll.DATA_DIR, "m2_v2_latency.json")
SEED = 42
N_TIMED, N_WARM = 2000, 200
BUDGET_MS = 2000.0


class ECIStep:
    """One-stream ECI recursion, one step at a time (same arithmetic as ll.eci_lag_sim, lag 0)."""
    def __init__(self, lr, alpha=0.1, T_burnin=5):
        self.lr, self.alpha, self.tb = lr, alpha, T_burnin
        self.q = 0.0; self.qt = 0.0; self.h = []

    def radius(self):
        return self.q

    def update(self, s, q_used):
        h = self.h
        lr_t = self.lr * (max(h) - min(h)) if h else self.lr
        grad = self.alpha if q_used >= s else -(1.0 - self.alpha)
        d = s - q_used; e = math.exp(-abs(d))
        self.qt -= lr_t * grad
        self.q = self.qt + lr_t * d * e / ((1.0 + e) ** 2)
        h.append(s)
        if len(h) > self.tb:
            h.pop(0)


def pct(a):
    a = np.asarray(a)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max()), "n": int(len(a))}


def time_calls(fn, n, sync=False):
    out = np.empty(n)
    for i in range(n):
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(i)
        if sync: torch.cuda.synchronize()
        out[i] = (time.perf_counter() - t0) * 1000.0
    return out


def main():
    print("Loading v2 test tensors...")
    df_test = add_hybrid_static_features(load_split_frame("test_v2"))
    df_te = sort_chronologically(df_test)
    Xt, Mt, Dt, Pt, yt = prepare_temporal_tensors_v2(df_te, "test_v2", device=DEVICE)
    St = torch.tensor(prepare_static_features_v2(df_te), device=DEVICE)
    idx = np.random.RandomState(0).choice(len(df_te), N_TIMED + N_WARM, replace=False)

    enc = load_frozen_encoder_v2(SEED)
    cfg = _tuned("m2_v2_hybrid_head_tuned_config.json")
    head = load_fusion_head_v2("m2_hybrid_grud_head", SEED, static_hidden=cfg["static_hidden"],
                               static_dropout=cfg["static_dropout"], fusion_dropout=cfg["fusion_dropout"])
    enc.eval(); head.eval()

    res = {"hardware": {"platform": platform.platform(), "processor": platform.processor(),
                        "cpu_count": os.cpu_count(), "torch": torch.__version__, "python": platform.python_version(),
                        "torch_threads": torch.get_num_threads(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
           "methodology": {"batch_size": 1, "model": "Hybrid GRU-D v2 (seed 42)", "n_timed": N_TIMED, "n_warmup": N_WARM,
                           "budget_ms": BUDGET_MS}, "paths": {}}

    def forward_bench(device):
        e, h_ = copy.deepcopy(enc).to(device), copy.deepcopy(head).to(device)   # separate copies so GPU and CPU paths never share moved modules
        e.eval(); h_.eval()
        tens = [t.to(device) for t in (Xt, Mt, Dt, Pt, St)]
        def call(i):
            j = int(idx[i])
            with torch.no_grad():
                hh = e(tens[0][j:j + 1], tens[1][j:j + 1], tens[2][j:j + 1], tens[3][j:j + 1])
                out = torch.sigmoid(h_(hh, tens[4][j:j + 1]))
            return out
        sync = device.type == "cuda"
        for i in range(N_WARM): call(i)
        return time_calls(lambda i: call(N_WARM + i), N_TIMED, sync=sync), call

    if torch.cuda.is_available():
        gpu_t, gpu_call = forward_bench(torch.device("cuda"))
        res["paths"]["forward_gpu"] = pct(gpu_t)
    cpu_t, _ = forward_bench(torch.device("cpu"))
    res["paths"]["forward_cpu"] = pct(cpu_t)
    print("forward GPU:", res["paths"].get("forward_gpu")); print("forward CPU:", res["paths"]["forward_cpu"])

    # ---- conformal paths over the full test stream (probabilities from the prediction cache)
    preds, y, _ = ll.get_all_predictions()
    p = preds[(ll.PRIMARY_MODEL, SEED)]
    sp = np.concatenate([p["fc"], p["te"]]).astype(float); sy = np.concatenate([y["fc"], y["te"]]).astype(int)
    n_cal, n = len(p["fc"]), len(sp)
    sc = np.abs(sy - sp); lr = ll.select_lr_marginal(p["cs"], y["cs"], 0)

    # verify ECIStep == simulator before timing it
    ref = ll.eci_lag_sim(sc, np.zeros(n, int), 1, 0, n_cal, lr)[:, 0]
    st = ECIStep(lr); mine = np.empty(n)
    for t in range(n):
        q = st.radius(); mine[t] = q; st.update(float(sc[t]), q)
    assert np.max(np.abs(ref - mine)) < 1e-9, "ECIStep diverges from the verified simulator"
    print("ECIStep == verified simulator: PASS")

    thr = 0.5
    t0 = time.perf_counter()
    for t in range(n_cal, n):
        _ = (abs(0 - sp[t]) <= thr, abs(1 - sp[t]) <= thr)
    res["paths"]["split_conformal_set"] = {"amortized_ms_per_arrival": (time.perf_counter() - t0) * 1000 / (n - n_cal)}

    st = ECIStep(lr); t0 = time.perf_counter()
    for t in range(n):
        q = st.radius(); _ = (abs(0 - sp[t]) <= q, abs(1 - sp[t]) <= q); st.update(float(sc[t]), q)
    res["paths"]["eci_step_plus_set"] = {"amortized_ms_per_arrival": (time.perf_counter() - t0) * 1000 / n}

    s0, s1 = ECIStep(lr), ECIStep(lr); t0 = time.perf_counter()
    for t in range(n):
        a, b = s0.radius(), s1.radius(); _ = (abs(0 - sp[t]) <= a, abs(1 - sp[t]) <= b)
        (s1 if sy[t] == 1 else s0).update(float(sc[t]), b if sy[t] == 1 else a)
    res["paths"]["mondrian_eci_step_plus_set"] = {"amortized_ms_per_arrival": (time.perf_counter() - t0) * 1000 / n}

    for W in (10000, 40000):
        w = sp[n_cal - min(W, n_cal):n_cal]; yy = sy[n_cal - min(W, n_cal):n_cal]
        ts = []
        for _ in range(50):
            t0 = time.perf_counter(); ll._class_threshold(np.abs(yy - w), 0.1); ts.append((time.perf_counter() - t0) * 1000)
        res["paths"][f"recalibration_refresh_W{W}"] = {"refresh_ms_median": float(np.median(ts)),
                                                        "amortized_ms_per_arrival": float(np.median(ts)) / ll.RECAL_R,
                                                        "window_used": int(len(w))}

    # ---- end-to-end: GPU forward + ECI step + set
    if torch.cuda.is_available():
        st = ECIStep(lr); ys = sy[n_cal:]
        def e2e(i):
            j = int(idx[N_WARM + i])
            pr_ = float(gpu_call(N_WARM + i).item())
            q = st.radius(); _ = (abs(0 - pr_) <= q, abs(1 - pr_) <= q); st.update(abs(float(ys[j]) - pr_), q)
        res["paths"]["end_to_end_gpu_forward_plus_eci"] = pct(time_calls(e2e, N_TIMED, sync=True))
        print("end-to-end:", res["paths"]["end_to_end_gpu_forward_plus_eci"])

    worst = max(v.get("p99_ms", v.get("max_ms", 0)) for v in res["paths"].values() if "p99_ms" in v)
    res["budget_check"] = {"budget_ms": BUDGET_MS, "worst_p99_ms_any_timed_path": worst, "within_budget": worst < BUDGET_MS}
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["paths"], indent=1)); print(f"budget check: {res['budget_check']}\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
