"""
M2 -- mTAN matched-rigor tuning pass (closes Deviation 15)
==============================================================================
m2_mtan_tuning_v2.py gave mTAN a deliberately lighter search than GRU-D and
XGBoost received (1 tune seed, 2 stages, 7 fits, selection by bare validation
AUROC, no noise guard, no calibration guard, no explicit untuned base) --
disclosed as Deviation 15 and left as a standing reviewer target. This script
gives mTAN the SAME protocol m2_xgboost_tuning_v2.py / m2_grud_tuning.py used:

  * 3 tune seeds [42, 123, 7]
  * staged, sequential search; each stage's candidates are overrides on the
    CURRENT best config, and the explicit UNTUNED_BASE is the origin (the
    Deviation-8 anti-drift guard)
  * noise guard: a candidate replaces the incumbent only if it beats the
    incumbent's mean validation AUROC by MORE than the incumbent's own
    validation SD (ddof=1) -- identical to pick_winner() in the XGBoost tuner
  * calibration guard: a candidate is rejected if its validation ECE exceeds
    5x the incumbent's
  * per-candidate AUROC / AUPRC / Brier / ECE / mean-prediction logging

Evaluated ONLY on the fixed dev-pool `validation` subject subset; test_v2 is
never loaded here. Seeds vary initialisation and batch order on the FIXED
dev-pool split, matching m2_hybrid_head_tuning_v2.py (the XGBoost tuner used a
random inner carve instead; that difference is disclosed there and is not
reproduced here because the GRU-D family and mTAN share the fixed split).

Stages (mTAN's analogues of XGBoost's five):
  1 capacity            (nhidden, embed_time)
  2 reference-time grid num_ref_points
  3 attention heads     num_heads
  4 optimisation        lr, weight_decay
  (5 class imbalance: NOT APPLICABLE -- the shared training loop
   _minibatch_early_stop_loop uses plain BCE with no class weighting for ANY v2
   neural model, so there is no pos_weight knob to tune; stated rather than
   silently dropped.)

Disclosed efficiency choice: the incumbent's result is REUSED as the next
stage's baseline instead of being re-fit (the XGBoost tuner re-evaluates its
baseline every stage). The incumbent was measured on exactly these seeds, so
reuse gives the identical comparison at ~1/3 fewer fits. ~13 evaluations x 3
seeds = 39 fits.

`--smoke` runs the whole pipeline on 4,000 rows for 2 epochs on 1 seed, to
verify plumbing (multi-head, lr/wd) in about a minute; it writes only
*_SMOKE files and never the real config.
"""
import os, sys, json, time, warnings, argparse
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MTAN_SRC = os.path.join(SCRIPT_DIR, "..", "..", "mTAN", "src")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, MTAN_SRC)
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from m2_hybrid_common import _minibatch_early_stop_loop, expected_calibration_error, DEVICE
from m2_v2_common import load_split_frame, load_dev_pool_split_v2
from m2_mtan_v2 import MTANClassifierHead, prepare_mtan_tensors, BATCH_SIZE, MAX_EPOCHS, PATIENCE

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
ARGS = parser.parse_args()
assert not any("test_v2" in str(a) for a in sys.argv), "This script must never touch test_v2."

SMOKE = ARGS.smoke
TUNE_SEEDS = [42] if SMOKE else [42, 123, 7]
RUN_MAX_EPOCHS = 2 if SMOKE else MAX_EPOCHS
RUN_PATIENCE = 2 if SMOKE else PATIENCE
TAG = "_SMOKE" if SMOKE else ""
TRACE_CSV = os.path.join(DATA_DIR, f"m2_v2_mtan_tuning_trace_matched{TAG}.csv")
CONFIG_JSON = os.path.join(DATA_DIR, f"m2_v2_mtan_tuned_config_matched{TAG}.json")

# The untuned reference: m2_mtan_v2.py's defaults, stated explicitly so no stage's
# baseline can drift.
UNTUNED_BASE = {"nhidden": 32, "embed_time": 32, "num_ref_points": 16, "num_heads": 1,
                "lr": 1e-3, "weight_decay": 1e-4}

trace = []


def fit_eval(cfg, seed, data):
    Xf, Tf, yf, Xv, Tv, yv = data
    torch.manual_seed(seed)
    model = MTANClassifierHead(
        input_dim=7, query=torch.linspace(0, 1, cfg["num_ref_points"]), nhidden=cfg["nhidden"],
        embed_time=cfg["embed_time"], num_heads=cfg["num_heads"], device=DEVICE).to(DEVICE)
    t0 = time.perf_counter()
    hist = _minibatch_early_stop_loop(
        param_groups_fn=lambda: model.parameters(),
        forward_fn=lambda ts: model(ts[0], ts[1]),
        X_fit_tensors=[Xf, Tf], y_fit=yf.cpu().numpy(), X_val_tensors=[Xv, Tv], y_val=yv.cpu().numpy(),
        seed=seed, lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=BATCH_SIZE,
        max_epochs=RUN_MAX_EPOCHS, patience=RUN_PATIENCE, train_modules=[model])
    secs = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xv, Tv)).cpu().numpy().ravel()
    y = yv.cpu().numpy()
    return {"val_auroc": roc_auc_score(y, p), "val_auprc": average_precision_score(y, p),
            "val_brier": brier_score_loss(y, p), "val_ece": expected_calibration_error(y, p),
            "val_mean_pred": float(p.mean()), "epochs": len(hist), "seconds": secs}


def evaluate(stage, name, cfg, data):
    rows = []
    for seed in TUNE_SEEDS:
        r = fit_eval(cfg, seed, data)
        rows.append(r)
        trace.append({"stage": stage, "candidate": name, "seed": seed,
                      **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                      **{f"cfg_{k}": v for k, v in cfg.items()}})
    pd.DataFrame(trace).to_csv(TRACE_CSV, index=False)  # incremental: an interruption loses nothing
    a = np.array([r["val_auroc"] for r in rows])
    mean, sd = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
    ece = float(np.mean([r["val_ece"] for r in rows]))
    print(f"  [{stage}] {name:<34} val AUROC={mean:.4f}+/-{sd:.4f}  "
          f"AUPRC={np.mean([r['val_auprc'] for r in rows]):.4f}  ECE={ece:.4f}  "
          f"mean_p={np.mean([r['val_mean_pred'] for r in rows]):.4f}  "
          f"({np.mean([r['seconds'] for r in rows]):.0f}s/fit)", flush=True)
    return mean, sd, ece


def run_stage(stage, title, incumbent_cfg, incumbent_res, candidates, data, header=True):
    """candidates: list of (name, overrides). Returns (winner_cfg, winner_res, winner_name)."""
    if header:
        print("\n" + "=" * 78 + f"\nSTAGE {stage}: {title}\n" + "=" * 78)
    base_mean, base_sd, base_ece = incumbent_res
    best_name, best_cfg, best_res = "baseline", incumbent_cfg, incumbent_res
    for name, overrides in candidates:
        cfg = {**incumbent_cfg, **overrides}
        if cfg["embed_time"] % cfg["num_heads"] != 0:
            print(f"  [{stage}] {name:<34} SKIPPED (embed_time {cfg['embed_time']} not divisible by heads {cfg['num_heads']})")
            continue
        res = evaluate(stage, name, cfg, data)
        mean, sd, ece = res
        if ece > 5 * max(base_ece, 1e-6):
            print(f"      -> {name} REJECTED by calibration guard (ECE {ece:.4f} vs incumbent {base_ece:.4f})")
            continue
        if mean - base_mean > base_sd:                      # noise guard
            if best_name == "baseline" or mean > best_res[0]:
                best_name, best_cfg, best_res = name, cfg, res
    print(f"  -> Stage {stage} winner: {best_name} (val AUROC {best_res[0]:.4f}, incumbent {base_mean:.4f} +/- {base_sd:.4f})")
    return best_cfg, best_res, best_name


def main():
    print(f"Device: {DEVICE}  smoke={SMOKE}  tune_seeds={TUNE_SEEDS}")
    df_train = load_split_frame("train_v2")
    dev = load_dev_pool_split_v2()
    tr = set(dev["training"]["subject_ids"].tolist()); va = set(dev["validation"]["subject_ids"].tolist())
    df_fit = df_train[df_train["subject_id"].isin(tr)]
    df_val = df_train[df_train["subject_id"].isin(va)]
    if SMOKE:
        df_fit, df_val = df_fit.iloc[:4000], df_val.iloc[:2000]
    Xf, Tf, yf = prepare_mtan_tensors(df_fit, "train_v2", DEVICE)
    Xv, Tv, yv = prepare_mtan_tensors(df_val, "train_v2", DEVICE)
    print(f"fit={len(df_fit)} ({int(yf.sum())} pos)  val={len(df_val)} ({int(yv.sum())} pos)")
    data = (Xf, Tf, yf, Xv, Tv, yv)

    # Stage 1: the baseline is measured fresh (it is also its own candidate); the other four
    # (nhidden, embed_time) pairs are then screened against it under the same guards.
    print("\n" + "=" * 78 + "\nSTAGE 1_capacity: capacity (nhidden, embed_time)\n" + "=" * 78)
    base_res = evaluate("1_capacity", "defaults (32,32)", UNTUNED_BASE, data)
    inc_cfg, inc_res, _ = run_stage("1_capacity", "capacity", dict(UNTUNED_BASE), base_res, [
        ("(16,16)", {"nhidden": 16, "embed_time": 16}), ("(64,64)", {"nhidden": 64, "embed_time": 64}),
        ("(32,16)", {"embed_time": 16}), ("(32,64)", {"embed_time": 64})], data, header=False)

    inc_cfg, inc_res, _ = run_stage("2_refgrid", "reference-time grid", inc_cfg, inc_res,
                                    [("num_ref_points=8", {"num_ref_points": 8}), ("num_ref_points=32", {"num_ref_points": 32})], data)
    inc_cfg, inc_res, _ = run_stage("3_heads", "attention heads", inc_cfg, inc_res,
                                    [("num_heads=2", {"num_heads": 2}), ("num_heads=4", {"num_heads": 4})], data)
    inc_cfg, inc_res, _ = run_stage("4_optim", "optimisation (lr, weight_decay)", inc_cfg, inc_res,
                                    [("lr=5e-4", {"lr": 5e-4}), ("lr=2e-3", {"lr": 2e-3}),
                                     ("weight_decay=1e-3", {"weight_decay": 1e-3}), ("weight_decay=0", {"weight_decay": 0.0})], data)

    winner = {**inc_cfg, "val_auroc": inc_res[0], "val_auroc_sd": inc_res[1], "val_ece": inc_res[2],
              "tune_seeds": TUNE_SEEDS,
              "protocol": "3-seed, 4-stage, noise-guarded (1 SD) + ECE-guarded (5x), UNTUNED_BASE origin; "
                          "stage 5 (class imbalance) not applicable to the shared BCE loop",
              "smoke": SMOKE}
    with open(CONFIG_JSON, "w") as f:
        json.dump(winner, f, indent=2)
    print(f"\n=== mTAN v2 matched-rigor config ===\n{json.dumps(winner, indent=2)}\nWrote {TRACE_CSV}, {CONFIG_JSON}")


if __name__ == "__main__":
    main()
