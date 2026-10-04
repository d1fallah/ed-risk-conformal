"""
M2 -- mTAN tuning pass, irregular-series (v2) cohort (Phase 5)
==============================================================================
Guards against the exact tuning-asymmetry trap this module already caught
once (M2_Findings_Dossier.md Sec 18.1: XGBoost beating an UNTUNED GRU-D is
not evidence of anything, and the module's own headline had to be
reframed once that was fixed). The pilot's single-seed, default-config
mTAN result (AUROC 0.8401) is not compared against GRU-D v2 as a paper
number until mTAN gets its own tuning pass under the same discipline.

Disclosed, deliberate scope reduction vs. m2_grud_tuning.py / m2_xgboost_
tuning.py's 3-seed, baseline-vs-SD-guarded staged search: this pass uses
a SINGLE tune seed (42) and a compact 2-stage grid (8 candidates total),
selected by validation AUROC, because it runs the night before an
unattended ~5h main grid and a full 3-seed x 15-candidate search (as GRU-D
and XGBoost received) would cost proportionally more of that budget than
a first comparator's tuning warrants. This is a real methodological
difference from the other models' tuning rigor -- documented here, in the
tuning-trace CSV, and in the dossier update, not silently equalized.
If mTAN's tuned result becomes load-bearing for the paper's headline
comparison, a proper 3-seed guarded re-tune is the recommended follow-up.

Stage 1: (nhidden, embed_time) in {(16,16),(32,32),(64,64),(32,16),(32,64)}
  -- nhidden=32 matches GRU-D v2's own hidden size, so this stage also
  answers "does giving mTAN GRU-D's own capacity change the outcome?"
Stage 2: num_ref_points in {8,16,32} at the Stage-1 winner's (nhidden,
  embed_time) -- the size of the fixed reference time grid mTAN attends
  onto; unlike GRU-D, this has no GRU-D-side analogue to match.

Evaluated ONLY on the fixed `validation` subject subset -- test.csv is
never touched here, per this project's anti-p-hacking convention
(m2_xgboost_tuning.py's own `pick_winner()` asserts it never opens
test.csv; this script honours the same rule even without importing that
exact assertion).
"""
import os, sys, json, time, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MTAN_SRC = os.path.join(SCRIPT_DIR, "..", "..", "mTAN", "src")
TRACE_CSV = os.path.join(DATA_DIR, "m2_v2_mtan_tuning_trace.csv")
CONFIG_JSON = os.path.join(DATA_DIR, "m2_v2_mtan_tuned_config.json")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, MTAN_SRC)
from sklearn.metrics import roc_auc_score
from m2_hybrid_common import _minibatch_early_stop_loop, DEVICE
from m2_v2_common import load_split_frame, load_dev_pool_split_v2
from m2_mtan_v2 import MTANClassifierHead, prepare_mtan_tensors, LR, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE

TUNE_SEED = 42


def evaluate_candidate(nhidden, embed_time, num_ref_points, num_heads, Xf, Tf, yf, Xv, Tv, yv):
    torch.manual_seed(TUNE_SEED)
    query = torch.linspace(0, 1, num_ref_points)
    model = MTANClassifierHead(
        input_dim=7, query=query, nhidden=nhidden, embed_time=embed_time,
        num_heads=num_heads, device=DEVICE,
    ).to(DEVICE)
    t0 = time.perf_counter()
    history = _minibatch_early_stop_loop(
        param_groups_fn=lambda: model.parameters(),
        forward_fn=lambda ts: model(ts[0], ts[1]),
        X_fit_tensors=[Xf, Tf], y_fit=yf.cpu().numpy(), X_val_tensors=[Xv, Tv], y_val=yv.cpu().numpy(),
        seed=TUNE_SEED, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, train_modules=[model],
    )
    elapsed = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(Xv, Tv)).cpu().numpy().ravel()
    val_auroc = roc_auc_score(yv.cpu().numpy(), val_prob)
    return {"nhidden": nhidden, "embed_time": embed_time, "num_ref_points": num_ref_points,
            "num_heads": num_heads, "val_auroc": float(val_auroc),
            "epochs": len(history), "seconds": elapsed}


def main():
    print(f"Device: {DEVICE}")
    df_train = load_split_frame("train_v2")
    dev_split = load_dev_pool_split_v2()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())
    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]

    Xf, Tf, yf = prepare_mtan_tensors(df_fit, "train_v2", DEVICE)
    Xv, Tv, yv = prepare_mtan_tensors(df_val, "train_v2", DEVICE)
    print(f"fit={len(df_fit)}, val={len(df_val)}")

    trace = []

    print("\n=== Stage 1: (nhidden, embed_time) ===")
    stage1_candidates = [(16, 16), (32, 32), (64, 64), (32, 16), (32, 64)]
    for nh, et in stage1_candidates:
        r = evaluate_candidate(nh, et, num_ref_points=16, num_heads=1,
                                Xf=Xf, Tf=Tf, yf=yf, Xv=Xv, Tv=Tv, yv=yv)
        r["stage"] = 1
        trace.append(r)
        print(f"  nhidden={nh} embed_time={et}: val_auroc={r['val_auroc']:.4f} "
              f"epochs={r['epochs']} ({r['seconds']:.0f}s)")

    best1 = max([r for r in trace if r["stage"] == 1], key=lambda r: r["val_auroc"])
    print(f"Stage 1 winner: nhidden={best1['nhidden']} embed_time={best1['embed_time']} "
          f"(val_auroc={best1['val_auroc']:.4f})")

    print("\n=== Stage 2: num_ref_points ===")
    for nrp in (8, 16, 32):
        if nrp == 16:  # already evaluated as part of Stage 1's baseline num_ref_points
            r = dict(best1)
            r["num_ref_points"] = 16
        else:
            r = evaluate_candidate(best1["nhidden"], best1["embed_time"], num_ref_points=nrp,
                                    num_heads=1, Xf=Xf, Tf=Tf, yf=yf, Xv=Xv, Tv=Tv, yv=yv)
        r["stage"] = 2
        trace.append(r)
        print(f"  num_ref_points={nrp}: val_auroc={r['val_auroc']:.4f}")

    best2 = max([r for r in trace if r["stage"] == 2], key=lambda r: r["val_auroc"])
    winner = {"nhidden": best2["nhidden"], "embed_time": best2["embed_time"],
              "num_ref_points": best2["num_ref_points"], "num_heads": best2["num_heads"],
              "val_auroc": best2["val_auroc"],
              "tune_seed": TUNE_SEED, "protocol": "1-seed, 2-stage, 8-candidate (disclosed scope reduction -- see module docstring)"}

    import pandas as pd
    pd.DataFrame(trace).to_csv(TRACE_CSV, index=False)
    with open(CONFIG_JSON, "w") as f:
        json.dump(winner, f, indent=2)

    print(f"\n=== mTAN v2 tuned config ===\n{json.dumps(winner, indent=2)}")
    print(f"Wrote {TRACE_CSV}, {CONFIG_JSON}")


if __name__ == "__main__":
    main()
