# M2 — Vendored-Baseline Provenance

Written retroactively (2026-09-05), during a repository-wide compliance audit, because no
provenance document existed for M2 — unlike `module 4/ncc_provenance.md` and
`module 5/informarl_provenance.md`, which this file follows in structure. Per `CLAUDE.md` §4
Step 1 ("Vendor the baseline repo(s)... record real numbers") and §1.10 (verify every
reference), this records what is actually on disk, checked live via `git log`/`git diff`/
`git remote -v` in each vendored tree, not copied from a web page or from memory.

## `mimic4ed-benchmark-main` — primary baseline (GRU-D pipeline, NEWS2)

- **Roadmap URL**: https://github.com/nliulab/mimic4ed-benchmark
- **Local form**: an **extracted ZIP**, not a git clone — there is no `.git` directory inside
  `mimic4ed-benchmark-main/`. The `-main` suffix is GitHub's default zip-download naming.
- **Commit hash: UNRECOVERABLE.** Because the tree has no git history, there is no way to
  determine which upstream commit it was extracted from, and no way to diff M2's own additions
  against the unmodified baseline. This is a real gap, not a stylistic one — module 4's
  equivalent provenance file records three commit hashes recoverable by a live `git log -1`;
  this repository cannot produce that for `mimic4ed-benchmark-main`.
- **What is M2's own work vs. upstream**: M2's ~30 `m2_*.py` scripts, `train_grud_conformal.py`,
  and `generate_m2_report.py` are net-new files, distinguishable from upstream by name and by
  the fact that upstream's own files (`helpers.py`, `medcode_utils.py`, the `Task_*.ipynb`
  notebooks) carry an October 2022 timestamp in the original repository's own history, while
  every `m2_*` file was authored for this project. This naming convention is the only provenance
  boundary available, in the absence of a real git history.
- **One upstream file WAS modified**: `Benchmark_scripts/helpers.py`'s `add_score_NEWS2()` —
  documented in `mimic4ed-benchmark-main/M2_Hybrid_Deviation_Log.md`, Deviation 10 (the vendored
  NEWS2 implementation scored only 4 of its 7 RCP parameters; the SpO2 sub-score was restored).
  This is a disclosed, justified correctness fix, but because there is no version-controlled
  baseline to diff against, the exact before/after change cannot be reproduced from git alone —
  only from that deviation log's own prose description.
- **Recommendation for a future session**: if the exact upstream commit can still be identified
  (e.g., from a download timestamp or a cached copy), re-clone it as a real `.git` checkout
  alongside the existing extracted tree, so this gap can finally close.

## `Error-quantified-Conformal-Inference` — ECI (ICLR 2025), the primary research contribution

- **URL**: https://github.com/creator-xi/Error-quantified-Conformal-Inference
- **Local form**: a real git clone, `.git` retained.
- **Commit**: `05086e774e9c8836cae299774549abda14b0a290` (2025-02-02, "Add files via upload"),
  confirmed via a live `git log -1` inside the clone.
- **Modification — environment-compatibility only, not previously documented.** `git diff`
  inside the clone shows two files patched:

  | File | Change |
  | :--- | :--- |
  | `core/methods.py` | 10 occurrences of `np.infty` → `np.inf` across `trailing_window`, `aci_clipped` (×3), `aci` (×2), `mytan` (×2), `quantile_integrator_log_scorecaster` (1, inside a comment) |
  | `core/quantile.py` | `standard_weighted_quantile`'s `maxscore=np.infty` default → `np.inf` |

  `np.infty` was removed as a NumPy alias; `np.inf` is the exact value it always meant, so this
  changes no behavior — the same classification module 5's `informarl_compat_shims.py` uses for
  its own numpy-alias restorations (CLAUDE.md §1.7: environment-compatibility fixes vs.
  methodological modifications). It was not previously written down anywhere in M2's own
  documentation (`M2_Hybrid_Deviation_Log.md`'s 11 entries cover NEWS2, GRU-D decay, scaling,
  ONNX export, and test-set leakage, but not this), and `README.md`'s "vendored repositories...
  not modified" line and the module's `.gitignore` comment both contradicted it until this audit.
  ECI keeps its own nested `.git`, so this patch is not tracked as ordinary files from the outer
  `module2` repository; the full diff is reproduced below so it stays auditable regardless:

  ```diff
  diff --git a/core/methods.py b/core/methods.py
  index 5daac87..88b8df7 100644
  --- a/core/methods.py
  +++ b/core/methods.py
  @@ -26,7 +26,7 @@ def trailing_window(
       for t in tqdm(range(T_test)):
           t_pred = t - ahead + 1
           if min(weight_length, t_pred) < np.ceil(1/alpha):
  -            qs[t] = np.infty
  +            qs[t] = np.inf
           else:
               qs[t] = np.quantile(scores[max(t_pred-weight_length,0):t_pred], 1-alpha, method='higher')
       results = {"method": "Trail", "q" : qs}
  @@ -49,11 +49,11 @@ def aci_clipped(
       covereds = np.zeros((T_test,))
       for t in tqdm(range(T_test)):
           t_pred = t - ahead + 1
  -        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.infty
  +        clip_value = scores[max(t_pred-window_length,0):t_pred].max() if t_pred > 0 else np.inf
           if t_pred > T_burnin:
               # Setup: current gradient
               if alphat <= 1/(t_pred+1):
  -                qs[t] = np.infty
  +                qs[t] = np.inf
               else:
                   qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
               covereds[t] = qs[t] >= scores[t]
  @@ -66,8 +66,8 @@ def aci_clipped(
               if t_pred > np.ceil(1/alpha):
                   qs[t] = np.quantile(scores[:t_pred], 1-alpha)
               else:
  -                qs[t] = np.infty
  -        if qs[t] == np.infty:
  +                qs[t] = np.inf
  +        if qs[t] == np.inf:
               qs[t] = clip_value
       results = { "method": "ACI (clipped)", "q" : qs, "alpha" : alphas}
       return results
  @@ -93,7 +93,7 @@ def aci(
           if t_pred > T_burnin:
               # Setup: current gradient
               if alphat <= 1/(t_pred+1):
  -                qs[t] = np.infty
  +                qs[t] = np.inf
               else:
                   qs[t] = np.quantile(scores[max(t_pred-window_length,0):t_pred], 1-np.clip(alphat, 0, 1), method='higher')
               covereds[t] = qs[t] >= scores[t]
  @@ -106,7 +106,7 @@ def aci(
               if t_pred > np.ceil(1/alpha):
                   qs[t] = np.quantile(scores[:t_pred], 1-alpha)
               else:
  -                qs[t] = np.infty
  +                qs[t] = np.inf
       results = { "method": "ACI", "q" : qs, "alpha" : alphas}
   
       return results
  @@ -131,9 +131,9 @@ def quantile(
   
   def mytan(x):
       if x >= np.pi/2:
  -        return np.infty
  +        return np.inf
       elif x <= -np.pi/2:
  -        return -np.infty
  +        return -np.inf
       else:
           return np.tan(x)
   
  @@ -316,7 +316,7 @@ def quantile_integrator_log_scorecaster(
           #integrator = saturation_fn_log((1-covereds)[T_burnin:t_pred].sum() - (t_pred-T_burnin)*alpha, (t_pred-T_burnin), Csat, KI) if t_pred > T_burnin else 0
           integrator_arg = (1-covereds)[:t_pred].sum() - (t_pred)*alpha
           #if onesided_integrator:
  -        #    integrator_arg = np.clip(integrator_arg, 0, np.infty)
  +        #    integrator_arg = np.clip(integrator_arg, 0, np.inf)
           integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI)
           # Train and scorecast if necessary
           if scorecast and train_model and t_pred > T_burnin and t+ahead < T_test:
  diff --git a/core/quantile.py b/core/quantile.py
  index f69604e..f159a25 100644
  --- a/core/quantile.py
  +++ b/core/quantile.py
  @@ -3,7 +3,7 @@ from scipy.optimize import brentq
   import pdb
   
   # Quantile function
  -def standard_weighted_quantile(scores, wtildes, quantile, maxscore=np.infty, allow_illegal_weights=False):
  +def standard_weighted_quantile(scores, wtildes, quantile, maxscore=np.inf, allow_illegal_weights=False):
       if not allow_illegal_weights:
           assert (np.array(wtildes).size == 0) or (wtildes.sum() <= 1.0 + 1e-4)
       if scores.shape[0] <= 5: # If not enough data, return infinite set
  ```

  No other modification is made to the vendored ECI codebase — no hyperparameter, algorithm, or
  calibration-logic change of any kind.

## `MDS-ED` — reference only, not imported into any results path

- **URL**: https://github.com/AI4HealthUOL/MDS-ED
- **Local form**: real git clone, `.git` retained, working tree clean (`git status` reports no
  modifications).
- **Commit**: `cbf408557f57d6f8424333273b72431b63cbb2b1` (2025-04-30).
- Consulted as a reference per the roadmap ("consult MDS-ED... as references only"); not
  imported by any M2 script.

## `CAREBench` — reference only, not imported into any results path

- **URL**: https://github.com/SeewonChoi/CAREBench
- **Local form**: real git clone, `.git` retained, working tree clean.
- **Commit**: `d53d9b5849b5c5e0c22e1360a5b7a4a1b82a7b0c` (2026-02-05).
- Consulted as a reference per the roadmap; not imported by any M2 script.

## `MAPIE` — pinned dependency, not cloned

- Pinned `mapie==1.5.0` in `mimic4ed-benchmark-main/requirements.txt`. This is compliant with
  `CLAUDE.md` §4 Step 1, which permits vendoring "into `/vendor/` **or as pinned dependencies**."
  No modification.

## Summary table

| Repo | Form | Commit | Modified? |
| :--- | :--- | :--- | :--- |
| mimic4ed-benchmark-main | extracted zip, no `.git` | **unrecoverable** | Yes — `helpers.py` (Deviation 10), undiffable |
| Error-quantified-Conformal-Inference | real clone | `05086e774e9c8836cae299774549abda14b0a290` | Yes — `core/methods.py`, `core/quantile.py`, env-compat only, diff above |
| MDS-ED | real clone | `cbf408557f57d6f8424333273b72431b63cbb2b1` | No |
| CAREBench | real clone | `d53d9b5849b5c5e0c22e1360a5b7a4a1b82a7b0c` | No |
| MAPIE | pip (`==1.5.0`) | n/a | No |
