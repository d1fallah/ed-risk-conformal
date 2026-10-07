"""
Phase 2.3 (neural models, 2 h landmark; R6) -- resumable.

  stage temporal       Temporal-Only GRU-D: tune (4 configs x tuning seeds {42,123,7}), then 5 final seeds (checkpoints + scores)
  stage hybrid_frozen  Hybrid GRU-D with the encoder of the SAME seed frozen from the 2 h Temporal-Only run (ablation, R6): tune the head
                       (4 configs x 3 tuning seeds), then 5 final seeds
  stage e2e            Hybrid GRU-D trained end-to-end on 2 h inputs (primary, R6): tune (4 configs x 3 tuning seeds), then 5 final seeds

Budget and search spaces: results/revision_v2/phase2/TUNING_BUDGET.md (written before any run). Selection metric: validation AUROC on the primary
outcome, mean over the 3 tuning seeds. Reuses the repository's GRU-D cell, masked runner, StaticEncoder, FusionHead and the shared mini-batch
early-stopping loop; nothing existing (checkpoints, predictions) is touched: outputs go to data/revision_v2/.
Each (stage, config, seed) writes a small JSON under data/revision_v2/runs/; reruns skip finished runs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rv2_common as C  # noqa: E402
import p2_common as P  # noqa: E402
from m2_hybrid_common import (_minibatch_early_stop_loop, DEVICE, StaticEncoder, FusionHead)  # noqa: E402
from m2_v2_common import (GRUDWithEncoderV2, TemporalEncoderV2, get_vitals_scaler_v2, prepare_static_features_v2,  # noqa: E402
                          VITALS_COLS)

GRUD_GRID = [(32, 1e-2), (64, 1e-2), (32, 3e-3), (64, 3e-3)]                 # (hidden, lr)
HEAD_GRID = [(32, 0.1), (32, 0.3), (64, 0.1), (64, 0.3)]                      # (static_hidden, static_dropout)
DROPOUT, WD, BATCH, MAX_EPOCHS, PATIENCE = 0.2, 1e-4, 128, 100, 10
E2E_STATIC_HIDDEN, E2E_STATIC_DROPOUT, FUSION_DROPOUT = 64, 0.3, 0.1
HEAD_LR = 1e-3


# ----------------------------------------------------------------------------------------------- data
def build_tensors():
    df = P.load_cohort("h2", "primary")
    sc = get_vitals_scaler_v2()
    mu = np.array([sc["mean"][c] for c in VITALS_COLS], dtype=np.float64)
    sd = np.array([sc["sd"][c] for c in VITALS_COLS], dtype=np.float64)
    out = {}
    v, o, p, de, th = P.h2_arrays(df)
    for r in P.ROLES:
        d = P.role(df, r)
        d_idx = df[df["split_role"] == r].index
        sel = d_idx.values
        X = np.where(o[sel].astype(bool), (v[sel] - mu) / sd, 0.0)
        out[r] = {"df": d, "y": d["y"].values.astype(np.float32),
                  "X": torch.tensor(X, dtype=torch.float32, device=DEVICE), "M": torch.tensor(o[sel], dtype=torch.float32, device=DEVICE),
                  "D": torch.tensor(de[sel], dtype=torch.float32, device=DEVICE), "P": torch.tensor(p[sel], dtype=torch.float32, device=DEVICE),
                  "S": torch.tensor(prepare_static_features_v2(d), dtype=torch.float32, device=DEVICE)}
    del v, o, p, de, th
    return out


def batched(fn, T, bs=20000):
    outs = []
    n = T["X"].shape[0]
    with torch.no_grad():
        for i in range(0, n, bs):
            outs.append(fn(T["X"][i:i + bs], T["M"][i:i + bs], T["D"][i:i + bs], T["P"][i:i + bs], T["S"][i:i + bs]))
    return torch.cat(outs).cpu().numpy().ravel()


def val_auroc(fn, T):
    return float(roc_auc_score(T["y"], batched(fn, T)))


def run_path(stage, cfg_i, seed, tag):
    return os.path.join(P.RUN_DIR, f"{stage}_{tag}_c{cfg_i}_s{seed}.json")


def done(path):
    return os.path.exists(path)


# ----------------------------------------------------------------------------------------------- models
def train_temporal(T, hid, lr, seed):
    torch.manual_seed(seed)
    model = GRUDWithEncoderV2(inp=7, hid=hid, stat=0, clf_hidden=32, dropout=DROPOUT).to(DEVICE)
    t0 = time.perf_counter()
    hist = _minibatch_early_stop_loop(
        param_groups_fn=lambda: model.parameters(), forward_fn=lambda ts: model(ts[0], ts[1], ts[2], ts[3]),
        X_fit_tensors=[T["fit"]["X"], T["fit"]["M"], T["fit"]["D"], T["fit"]["P"]], y_fit=T["fit"]["y"],
        X_val_tensors=[T["val"]["X"], T["val"]["M"], T["val"]["D"], T["val"]["P"]], y_val=T["val"]["y"],
        seed=seed, lr=lr, weight_decay=WD, batch_size=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE, train_modules=[model])
    model.eval()
    fn = lambda X, M, D, Pm, S: torch.sigmoid(model(X, M, D, Pm))  # noqa: E731
    return model, fn, hist, time.perf_counter() - t0


class E2E(torch.nn.Module):
    def __init__(self, hid, static_hidden, static_dropout, fusion_dropout):
        super().__init__()
        self.enc = TemporalEncoderV2(inp=7, hid=hid)
        self.head = FusionHead(temporal_dim=hid, static_encoder=StaticEncoder(in_dim=4, hidden=static_hidden, dropout=static_dropout),
                               fusion_dropout=fusion_dropout)

    def forward(self, X, M, D, Pm, S):
        return self.head(self.enc(X, M, D, Pm), S)


def train_e2e(T, hid, lr, seed):
    torch.manual_seed(seed)
    model = E2E(hid, E2E_STATIC_HIDDEN, E2E_STATIC_DROPOUT, FUSION_DROPOUT).to(DEVICE)
    t0 = time.perf_counter()
    hist = _minibatch_early_stop_loop(
        param_groups_fn=lambda: model.parameters(), forward_fn=lambda ts: model(ts[0], ts[1], ts[2], ts[3], ts[4]),
        X_fit_tensors=[T["fit"][k] for k in ("X", "M", "D", "P", "S")], y_fit=T["fit"]["y"],
        X_val_tensors=[T["val"][k] for k in ("X", "M", "D", "P", "S")], y_val=T["val"]["y"],
        seed=seed, lr=lr, weight_decay=WD, batch_size=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE, train_modules=[model])
    model.eval()
    fn = lambda X, M, D, Pm, S: torch.sigmoid(model(X, M, D, Pm, S))  # noqa: E731
    return model, fn, hist, time.perf_counter() - t0


def train_frozen_head(T, enc, static_hidden, static_dropout, seed):
    from m2_v2_common import train_on_frozen_encoder_v2
    head, hist = train_on_frozen_encoder_v2(
        enc, T["fit"]["X"], T["fit"]["M"], T["fit"]["D"], T["fit"]["P"], T["fit"]["y"], T["val"]["X"], T["val"]["M"], T["val"]["D"], T["val"]["P"],
        T["val"]["y"], seed=seed, S_fit=T["fit"]["S"], S_val=T["val"]["S"], static_hidden=static_hidden, static_dropout=static_dropout,
        fusion_dropout=FUSION_DROPOUT, lr=HEAD_LR, batch_size=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    head.eval()
    fn = lambda X, M, D, Pm, S: torch.sigmoid(head(enc(X, M, D, Pm), S))  # noqa: E731
    return head, fn, hist


def encoder_from_temporal(model, hid):
    enc = TemporalEncoderV2(inp=7, hid=hid).to(DEVICE)
    missing, _ = enc.load_state_dict(model.state_dict(), strict=False)
    assert not missing, f"encoder keys missing: {missing}"
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def store_scores(name, seed, T, fn):
    for r in P.ROLES:
        P.save_preds(f"h2_{name}_{r}", seed, T[r]["df"], batched(fn, T[r]))


# ----------------------------------------------------------------------------------------------- stages
def select(stage, grid):
    best = None
    traces = []
    for ci, cfg in enumerate(grid):
        aucs = []
        for s in C.TUNE_SEEDS:
            aucs.append(json.load(open(run_path(stage, ci, s, "tune")))["val_auroc"])
        traces.append({"config": list(cfg), "val_auroc_per_tune_seed": aucs, "val_auroc_mean": float(np.mean(aucs))})
        if best is None or traces[-1]["val_auroc_mean"] > traces[best]["val_auroc_mean"]:
            best = ci
    return best, traces


def stage_temporal(T):
    for ci, (hid, lr) in enumerate(GRUD_GRID):
        for s in C.TUNE_SEEDS:
            p = run_path("temporal", ci, s, "tune")
            if done(p):
                continue
            model, fn, hist, secs = train_temporal(T, hid, lr, s)
            json.dump({"hid": hid, "lr": lr, "seed": s, "val_auroc": val_auroc(fn, T["val"]), "epochs": len(hist), "seconds": secs}, open(p, "w"))
            print("temporal tune", ci, s, json.load(open(p)), flush=True)
            del model
    best, traces = select("temporal", GRUD_GRID)
    hid, lr = GRUD_GRID[best]
    sel = {"selected_index": best, "selected": {"hidden": hid, "lr": lr}, "tuning_trace": traces}
    json.dump(sel, open(os.path.join(P.RUN_DIR, "temporal_selected.json"), "w"), indent=2)
    for s in C.SEEDS:
        p = run_path("temporal", best, s, "final")
        if done(p):
            continue
        model, fn, hist, secs = train_temporal(T, hid, lr, s)
        torch.save(model.state_dict(), os.path.join(P.CKPT_DIR, f"h2_temporal_seed{s}.pt"))
        store_scores("temporal_grud", s, T, fn)
        json.dump({"hid": hid, "lr": lr, "seed": s, "val_auroc": val_auroc(fn, T["val"]), "epochs": len(hist), "seconds": secs}, open(p, "w"))
        print("temporal final", s, json.load(open(p)), flush=True)
        del model


def stage_hybrid_frozen(T):
    sel = json.load(open(os.path.join(P.RUN_DIR, "temporal_selected.json")))
    hid = sel["selected"]["hidden"]

    def encoder_for(seed):
        m = GRUDWithEncoderV2(inp=7, hid=hid, stat=0, clf_hidden=32, dropout=DROPOUT).to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(P.CKPT_DIR, f"h2_temporal_seed{seed}.pt"), map_location=DEVICE))
        return encoder_from_temporal(m, hid)

    for ci, (sh, sd) in enumerate(HEAD_GRID):
        for s in C.TUNE_SEEDS:
            p = run_path("hybrid_frozen", ci, s, "tune")
            if done(p):
                continue
            head, fn, hist = train_frozen_head(T, encoder_for(s), sh, sd, s)
            json.dump({"static_hidden": sh, "static_dropout": sd, "seed": s, "val_auroc": val_auroc(fn, T["val"]), "epochs": len(hist)}, open(p, "w"))
            print("hybrid_frozen tune", ci, s, json.load(open(p)), flush=True)
    best, traces = select("hybrid_frozen", HEAD_GRID)
    sh, sd = HEAD_GRID[best]
    json.dump({"selected_index": best, "selected": {"static_hidden": sh, "static_dropout": sd}, "encoder_hidden": hid, "tuning_trace": traces},
              open(os.path.join(P.RUN_DIR, "hybrid_frozen_selected.json"), "w"), indent=2)
    for s in C.SEEDS:
        p = run_path("hybrid_frozen", best, s, "final")
        if done(p):
            continue
        head, fn, hist = train_frozen_head(T, encoder_for(s), sh, sd, s)
        store_scores("hybrid_frozen", s, T, fn)
        json.dump({"seed": s, "val_auroc": val_auroc(fn, T["val"]), "epochs": len(hist)}, open(p, "w"))
        print("hybrid_frozen final", s, json.load(open(p)), flush=True)


def stage_e2e(T):
    for ci, (hid, lr) in enumerate(GRUD_GRID):
        for s in C.TUNE_SEEDS:
            p = run_path("e2e", ci, s, "tune")
            if done(p):
                continue
            model, fn, hist, secs = train_e2e(T, hid, lr, s)
            json.dump({"hid": hid, "lr": lr, "seed": s, "val_auroc": val_auroc(fn, T["val"]), "epochs": len(hist), "seconds": secs}, open(p, "w"))
            print("e2e tune", ci, s, json.load(open(p)), flush=True)
            del model
    best, traces = select("e2e", GRUD_GRID)
    hid, lr = GRUD_GRID[best]
    json.dump({"selected_index": best, "selected": {"hidden": hid, "lr": lr}, "tuning_trace": traces},
              open(os.path.join(P.RUN_DIR, "e2e_selected.json"), "w"), indent=2)
    for s in C.SEEDS:
        p = run_path("e2e", best, s, "final")
        if done(p):
            continue
        model, fn, hist, secs = train_e2e(T, hid, lr, s)
        torch.save(model.state_dict(), os.path.join(P.CKPT_DIR, f"h2_e2e_seed{s}.pt"))
        store_scores("hybrid_e2e", s, T, fn)
        json.dump({"hid": hid, "lr": lr, "seed": s, "val_auroc": val_auroc(fn, T["val"]), "epochs": len(hist), "seconds": secs}, open(p, "w"))
        print("e2e final", s, json.load(open(p)), flush=True)
        del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["temporal", "hybrid_frozen", "e2e"])
    a = ap.parse_args()
    print("stage", a.stage, "device", DEVICE, "started", time.strftime("%H:%M:%S"), flush=True)
    T = build_tensors()
    print({r: (len(T[r]["y"]), int(T[r]["y"].sum())) for r in P.ROLES}, flush=True)
    {"temporal": stage_temporal, "hybrid_frozen": stage_hybrid_frozen, "e2e": stage_e2e}[a.stage](T)
    print("stage", a.stage, "finished", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
