"""
Phase 2 sensitivity analysis (reviewer C1 for the recurrent models; run before any GPU re-run): are the stored GRU-D models distorted by the copy of the triage
measurement in the ED vital-sign table? INFERENCE ONLY, no refit, no change to any stored checkpoint or prediction.
The stored Hybrid GRU-D (end to end; primary) and Temporal-Only GRU-D checkpoints of the five final seeds are scored on the 2 h test stays after removing, for every stay
whose first ED vital-sign row repeats the triage measurement (same rule as p2_26: at least three of heart rate, respiratory rate, oxygen saturation, systolic and diastolic
pressure recorded in both, all equal), that row. The later rows move up one slot, and the elapsed-time input is rebuilt with the recursion of the repository builder
(Che et al. 2018, Eq. 8-9: time since the last observation of each variable, triage observed at hour 0), which is validated here against the stored tensors.
Decision rule fixed by the author before running (DEVIATIONS P2-30): if the seed-mean AUROC of Hybrid GRU-D (end to end) changes by 0.002 or more in either direction
(about two thirds of the C1 margin of 0.0033), the de-duplicated GRU-D refit is run; otherwise C1 stands with one added sentence.
Sanity checks: inference without any change reproduces the stored test predictions of each seed; rebuilt deltas reproduce the stored deltas of the unmodified tensors.
Temporal-Only is reported for completeness and is not part of the rule. Test AUROC with 1,000 patient-cluster bootstrap resamples (paired). Aggregates only.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402
import p2_3_neural as N  # noqa: E402
from p2_26_dedup_and_ablation_classical import drop_copied_first_row  # noqa: E402
from m2_hybrid_common import DEVICE  # noqa: E402
from m2_v2_common import GRUDWithEncoderV2, get_vitals_scaler_v2, prepare_static_features_v2, VITALS_COLS  # noqa: E402

RULE_THRESHOLD = 0.002


def rebuild_delta(obs, pad, th):
    """Elapsed-time input as in the repository builder: observed variable -> time since its last observation (triage at hour 0); unobserved -> previous delta plus the gap
    between consecutive timesteps; padded slots 0."""
    n, T, V = obs.shape
    delta = np.zeros((n, T, V), dtype=np.float64)
    seen = obs[:, 0, :] > 0
    last = np.zeros((n, V))
    prev_t = np.zeros(n)
    for t in range(1, T):
        cur = th[:, t].astype(np.float64)
        live = pad[:, t] > 0
        ob = (obs[:, t, :] > 0) & live[:, None]
        gap_last = np.where(seen, cur[:, None] - last, cur[:, None])
        gap = np.maximum(0.0, cur - prev_t)[:, None]
        d = np.where(ob, np.maximum(0.0, gap_last), delta[:, t - 1, :] + gap)
        delta[:, t, :] = np.where(live[:, None], d, 0.0)
        last = np.where(ob, cur[:, None], last)
        seen = seen | ob
        prev_t = np.where(live, cur, prev_t)
    return delta.astype(np.float32)


def tensors(values, obs, pad, delta, df):
    sc = get_vitals_scaler_v2()
    mu = np.array([sc["mean"][c] for c in VITALS_COLS], dtype=np.float64)
    sd = np.array([sc["sd"][c] for c in VITALS_COLS], dtype=np.float64)
    X = np.where(obs.astype(bool), (values - mu) / sd, 0.0)
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)  # noqa: E731
    return {"X": f(X), "M": f(obs), "D": f(delta), "P": f(pad), "S": f(prepare_static_features_v2(df))}


def load_model(kind, seed):
    if kind == "e2e":
        hid = json.load(open(os.path.join(P.RUN_DIR, "e2e_selected.json")))["selected"]["hidden"]
        m = N.E2E(hid, N.E2E_STATIC_HIDDEN, N.E2E_STATIC_DROPOUT, N.FUSION_DROPOUT).to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(P.CKPT_DIR, f"h2_e2e_seed{seed}.pt"), map_location=DEVICE))
        m.eval()
        return lambda T: N.batched(lambda X, M, D, Pm, S: torch.sigmoid(m(X, M, D, Pm, S)), T)
    hid = json.load(open(os.path.join(P.RUN_DIR, "temporal_selected.json")))["selected"]["hidden"]
    m = GRUDWithEncoderV2(inp=7, hid=hid, stat=0, clf_hidden=32, dropout=N.DROPOUT).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(P.CKPT_DIR, f"h2_temporal_seed{seed}.pt"), map_location=DEVICE))
    m.eval()
    return lambda T: N.batched(lambda X, M, D, Pm, S: torch.sigmoid(m(X, M, D, Pm)), T)


def main():
    df = P.load_cohort("h2", "primary")
    te = P.role(df, "test")
    y = te["y"].values.astype(int)
    v, o, p, de, th = P.h2_arrays(te)
    res = {"n_test": int(len(y)), "positives": int(y.sum()), "rule_threshold": RULE_THRESHOLD}
    # validation of the rebuilt elapsed-time input on the unmodified tensors
    rd = rebuild_delta(o, p, th)
    live = (p > 0)[:, :, None] & np.ones_like(o, bool)
    res["delta_rebuild_max_abs_diff_unmodified"] = float(np.abs(np.where(live, rd - de, 0.0)).max())
    res["delta_rebuild_share_within_1e-3"] = float(((np.abs(rd - de) < 1e-3) & live).sum() / live.sum())
    v2, o2, p2, de2, th2, flag = drop_copied_first_row(v, o, p, de, th)
    de2 = rebuild_delta(o2, p2, th2)
    res["flagged_share_test"] = float(flag.mean())
    T0 = tensors(v, o, p, de, te)
    T1 = tensors(np.nan_to_num(v2), o2, p2, de2, te)
    boot = C.ClusterBoot(te["subject_id"].values, B=1000)
    res["models"] = {}
    for kind, name in (("e2e", "hybrid_e2e"), ("temporal", "temporal_grud")):
        per = []
        for s in C.SEEDS:
            fn = load_model(kind, s)
            s0, s1 = fn(T0), fn(T1)
            sid, stored = P.load_preds(f"h2_{name}_test", s)
            assert np.array_equal(sid, te["stay_id"].values)
            stored = np.asarray(stored, float)
            pt0, d0 = C.boot_metric(y, stored, boot, "auroc")
            pt1, d1 = C.boot_metric(y, s1, boot, "auroc")
            per.append({"seed": s, "reproduction_max_abs_diff": float(np.abs(s0 - stored).max()), "auroc_stored": pt0, "auroc_dedup": pt1,
                        "change": pt1 - pt0, "change_ci95": C.pct_ci(d1 - d0), "_d": d1 - d0, "_a0": d0, "_a1": d1})
            print(kind, s, "repro diff", round(per[-1]["reproduction_max_abs_diff"], 6), "stored", round(pt0, 4), "dedup", round(pt1, 4),
                  "change", round(pt1 - pt0, 5), [round(x, 5) for x in per[-1]["change_ci95"]], flush=True)
        dm = np.mean([r["_d"] for r in per], axis=0)
        rec = {"per_seed": [{k: v_ for k, v_ in r.items() if not k.startswith("_")} for r in per],
               "auroc_stored_seed_mean": float(np.mean([r["auroc_stored"] for r in per])), "auroc_dedup_seed_mean": float(np.mean([r["auroc_dedup"] for r in per])),
               "change_seed_mean": float(np.mean([r["change"] for r in per])), "change_seed_mean_ci95": C.pct_ci(dm),
               "change_seed_sd": float(np.std([r["change"] for r in per], ddof=1)),
               "auroc_dedup_seed_mean_ci95": C.pct_ci(np.mean([r["_a1"] for r in per], axis=0)),
               "max_abs_change_over_seeds": float(max(abs(r["change"]) for r in per))}
        res["models"][kind] = rec
        print(kind, "seed-mean change", round(rec["change_seed_mean"], 5), [round(x, 5) for x in rec["change_seed_mean_ci95"]], flush=True)
    r = res["models"]["e2e"]
    res["rule_applied_to"] = "hybrid_e2e seed mean"
    res["refit_required"] = bool(abs(r["change_seed_mean"]) >= RULE_THRESHOLD)
    print("rule: |change| =", round(abs(r["change_seed_mean"]), 5), ">=", RULE_THRESHOLD, "->", res["refit_required"])
    C.dump(res, "phase2", "p2_27_grud_dedup_inference.json")


if __name__ == "__main__":
    main()
