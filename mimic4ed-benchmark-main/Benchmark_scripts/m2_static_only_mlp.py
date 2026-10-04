"""
M2 Hybrid Guide v3.8 -- Static-Only MLP (Sec 4.4, "Reuse Criteria")
========================================================================
Answers: "How much predictive value do static characteristics alone
provide?" Input is exactly the 4 Guide-mandated static features (age, sex,
triage acuity, Elixhauser comorbidity count) -- never the 24-feature
static_cols baked into the legacy 0.9009 checkpoint (see Sec 0's "Critical
rule" -- that checkpoint is not reused by anything in this experiment).

RETRAINED under v3.8 (was previously run under v3.2's per-seed-re-split
convention -- see M2_Findings_Dossier.md Sec 12 for that history). Sec 4.4's
reuse criteria require "same patient-level train/validation/test
partitions" as the new fixed-once 4-way split (Sec 5.1/Sec 7); the prior
run re-derived a fresh 90/10 train/val split *per seed*, which does not
satisfy that criterion, so it is retrained here rather than reused. Only the
partitioning mechanism changed -- architecture, grid, and epoch budget are
unchanged and still valid.

Per Guide Sec 8, conformal reporting (MAPIE/ECI, coverage/set-size) is
scoped to "Hybrid GRU-D, Temporal-Only GRU-D, and Frozen-Temporal Head
Control" -- Static-Only MLP reports discrimination (AUROC/AUPRC) and
probabilistic calibration (Brier/ECE/slope/intercept) only, matching Sec
10's final table (Coverage/Set-Size = "--" for this row).

Architecture (Sec 4.4/6.2): 4 inputs -> 1 hidden layer (grid: hidden in
{16,32,64}, dropout in {0.1,0.3}, Sec 6.2's "Static MLP Branch" grid --
Sec 4.4 explicitly requires "same tuning protocol (hidden dim {16,32,64} x
dropout {0.1,0.3})") -> ReLU -> Dropout -> 1 output logit -> sigmoid. Fixed
at 1 hidden layer per Sec 4.4's "one hidden layer" spec. Learning rate is
fixed at 1e-3 (Adam's common default, and one of the two values in Sec
6.3's fusion-head LR grid) since neither Sec 4.4 nor Sec 6.2 gives
Static-Only MLP its own LR dimension.

Light tuning protocol: each of the 6 (hidden, dropout) configs is evaluated
at 3 seeds (42, 123, 7 -- same TUNE_SEEDS convention as m2_grud_tuning.py)
on the FIXED `validation` subject subset (Sec 5.1's 4-way split, loaded via
m2_hybrid_common.build_fixed_dev_split() -- never re-derived); the config
with the highest mean validation AUROC is frozen (early stopping on
validation LOSS, patience=10, per Sec 6.5). Test set is never touched until
the final 5-seed confirmatory run.

Final protocol (Sec 4.4 point 6/Sec 7): 5 seeds {42,123,7,456,789}, using
the SAME fixed `training`/`validation` subject partition for every seed --
only weight initialization and data order vary by seed now, not which
patients are in which subset.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
SEED_DIR = os.path.join(DATA_DIR, "m2_hybrid_seed_results", "static_only_mlp")
TRACE_CSV = os.path.join(DATA_DIR, "m2_static_only_mlp_tuning_trace.csv")
os.makedirs(SEED_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from m2_hybrid_common import (
    TARGET, HYBRID_STATIC_COLS, add_hybrid_static_features,
    build_fixed_dev_split, expected_calibration_error, calibration_slope_intercept,
    save_predictions, save_seed_metrics, save_manifest,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TUNE_SEEDS = [42, 123, 7]
FINAL_SEEDS = [42, 123, 7, 456, 789]
GRID = [(h, d) for h in (16, 32, 64) for d in (0.1, 0.3)]
LR = 1e-3
# Diagnostic (see M2_Findings_Dossier.md Sec 12.1): full-batch Adam at lr=1e-3
# on this 4-feature input needs ~2000-2400 epochs to reach the point where
# validation loss actually plateaus (val AUROC still climbing from 0.32 at
# epoch 100 to 0.84 at epoch 2400) -- MAX_EPOCHS=100 was silently truncating
# every run mid-convergence, producing unstable, seed-dependent garbage
# (0.48-0.76 AUROC across 5 seeds) that never reflected a real early-stopping
# decision. 3000 gives headroom past the observed plateau at negligible cost
# (a 4-input MLP epoch is near-instant on this hardware).
MAX_EPOCHS = 3000
PATIENCE = 10


class StaticMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_one(X_tr, y_tr, X_val, y_val, hidden, dropout, seed, lr=LR,
              max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    model = StaticMLP(in_dim=X_tr.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    best_val_loss, patience_left, best_state = float("inf"), patience, None
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = patience
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best_state)
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(model(Xt)).cpu().numpy().ravel()


def main():
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    df_train = add_hybrid_static_features(df_train)
    df_test = add_hybrid_static_features(df_test)

    y_test = df_test[TARGET].astype(int).values
    subj_test = df_test["subject_id"].values
    stay_test = df_test["stay_id"].values

    # Fixed 4-way dev-pool split (Sec 5.1/Sec 7) -- built once, reused
    # identically by every seed and every model. Static-Only MLP only needs
    # the `training`/`validation` subsets (no conformal calibration here).
    dev_split = build_fixed_dev_split()
    train_ids = set(dev_split["training"]["subject_ids"].tolist())
    val_ids = set(dev_split["validation"]["subject_ids"].tolist())
    df_fit = df_train[df_train["subject_id"].isin(train_ids)]
    df_val = df_train[df_train["subject_id"].isin(val_ids)]
    X_fit_df = df_fit[HYBRID_STATIC_COLS].values
    X_val_df = df_val[HYBRID_STATIC_COLS].values
    y_fit = df_fit[TARGET].astype(int).values
    y_val = df_val[TARGET].astype(int).values
    print(f"Fixed partitions: training={len(df_fit)} rows / {len(train_ids)} patients, "
          f"validation={len(df_val)} rows / {len(val_ids)} patients")

    # ── Light tuning (validation only -- test.csv used only for the final
    # confirmatory run below, never during config selection) ───────────────
    print(f"\n{'='*60}\nLight tuning: {len(GRID)} configs x {len(TUNE_SEEDS)} seeds\n{'='*60}")
    trace_rows = []
    best_cfg, best_mean_auroc = None, -1.0
    for hidden, dropout in GRID:
        val_aurocs = []
        for seed in TUNE_SEEDS:
            scaler = StandardScaler().fit(X_fit_df)
            X_fit_s, X_val_s = scaler.transform(X_fit_df), scaler.transform(X_val_df)
            model = train_one(X_fit_s, y_fit, X_val_s, y_val, hidden, dropout, seed)
            p_val = predict_proba(model, X_val_s)
            auroc_val = roc_auc_score(y_val, p_val)
            val_aurocs.append(auroc_val)
            trace_rows.append({"hidden": hidden, "dropout": dropout, "seed": seed, "val_auroc": auroc_val})
        mean_auroc = float(np.mean(val_aurocs))
        print(f"  hidden={hidden:>2} dropout={dropout}: val AUROC = {mean_auroc:.4f} +/- {np.std(val_aurocs):.4f}")
        if mean_auroc > best_mean_auroc:
            best_mean_auroc, best_cfg = mean_auroc, (hidden, dropout)

    pd.DataFrame(trace_rows).to_csv(TRACE_CSV, index=False)
    hidden, dropout = best_cfg
    print(f"\nSelected config: hidden={hidden}, dropout={dropout} (val AUROC {best_mean_auroc:.4f})")
    print(f"Full trace saved to {TRACE_CSV}")

    # ── Final 5-seed confirmatory run (test set touched here only; the
    # training/validation SUBJECT PARTITION is now identical across all 5
    # seeds -- only weight init/data order vary, per Guide Sec 7) ──────────
    print(f"\n{'='*60}\nFinal 5-seed run at frozen config\n{'='*60}")
    for seed in FINAL_SEEDS:
        scaler = StandardScaler().fit(X_fit_df)
        X_fit_s = scaler.transform(X_fit_df)
        X_val_s = scaler.transform(X_val_df)
        X_test_s = scaler.transform(df_test[HYBRID_STATIC_COLS].values)

        model = train_one(X_fit_s, y_fit, X_val_s, y_val, hidden, dropout, seed)
        y_prob = predict_proba(model, X_test_s)

        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test, y_prob)
        slope, intercept = calibration_slope_intercept(y_test, y_prob)

        print(f"  seed={seed}: AUROC={auroc:.4f} AUPRC={auprc:.4f} Brier={brier:.4f} "
              f"ECE={ece:.4f} slope={slope:.3f} intercept={intercept:.3f}")

        result = {
            "seed": seed, "model": "Static-Only MLP",
            "config": {"hidden": hidden, "dropout": dropout, "lr": LR},
            "metrics": {
                "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier),
                "ece": float(ece), "calibration_slope": slope, "calibration_intercept": intercept,
            },
        }
        with open(os.path.join(SEED_DIR, f"seed_{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)

        save_predictions(subj_test, "Static-Only MLP", seed, y_test, y_prob, stay_id=stay_test)
        save_seed_metrics("Static-Only MLP", seed, result["metrics"])
        save_manifest("Static-Only MLP", seed, result["config"])

    # ── Console summary (persisted aggregation happens once all 5 Guide
    # models exist, in the blocked Phase F m2_hybrid_analyze.py) ───────────
    df_res = pd.read_csv(os.path.join(DATA_DIR, "results_hybrid.csv"))
    df_res = df_res[df_res["model"] == "Static-Only MLP"]
    print(f"\n5-seed summary (Static-Only MLP): "
          f"AUROC {df_res['auroc'].mean():.4f} +/- {df_res['auroc'].std():.4f}  "
          f"AUPRC {df_res['auprc'].mean():.4f} +/- {df_res['auprc'].std():.4f}  "
          f"Brier {df_res['brier'].mean():.4f} +/- {df_res['brier'].std():.4f}  "
          f"ECE {df_res['ece'].mean():.4f} +/- {df_res['ece'].std():.4f}")


if __name__ == "__main__":
    main()
