#!/usr/bin/env python3
"""
Single-command reproduction entrypoint for M2's headline results, per
CLAUDE.md Sec 1.3 and Sec 4 Step 6.

Two tracks, matching M2_Findings_Dossier.md's own structure:
  v2 (default) -- the irregular-series, chronological-primary-split
      pipeline (Sec 24), Gate v2 (signed off 2026-09-19 by Akram). This is
      what "headline result" means for M2 as of this session.
  t2 -- the superseded fixed-T=2, random-split pipeline (Secs 1-22),
      retained and independently reproducible from
      data/archive_pre_irregular/, never silently mixed with v2.

By default this starts from the already-extracted data/master_dataset.csv
(v2 track) or data/train.csv+test.csv (t2 track) -- both are large,
PHI-derived MIMIC-IV-ED artifacts that are gitignored, not committed, and
must exist locally from a prior extraction run. `--from-scratch` also runs
the raw-MIMIC-IV-ED extraction stage (extract_master_dataset.py), which
requires local access to a credentialed MIMIC-IV-ED download.

Does not create or manage a virtualenv -- unlike M1, M2's entire pipeline
(vendored mimic4ed-benchmark, TenSEAL is not involved here, MAPIE, ECI,
mTAN's PyTorch/vendored src) runs in one environment; this script uses
sys.executable throughout.

Usage:
    python experiments/run_reproducible_m2.py --config configs/paper_track_m2.yaml
    python experiments/run_reproducible_m2.py --config configs/paper_track_m2.yaml --track t2
    python experiments/run_reproducible_m2.py --config configs/paper_track_m2.yaml --from-scratch
    python experiments/run_reproducible_m2.py --config configs/paper_track_m2.yaml --stage hybrid_grud
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.join(_HERE, "..")
SCRIPTS = os.path.join(MODULE_ROOT, "Benchmark_scripts")

PY = sys.executable

# (stage name, script, [args]) in dependency order. Frozen-encoder stages
# (frozen_temporal_control*, hybrid_grud*) require ALL confirmatory seeds
# of the preceding temporal_only_grud* stage's checkpoints to already
# exist -- enforced by those scripts' own asserts, not by this
# orchestrator, so a `--stage` debug run of a downstream stage fails
# loudly rather than silently using a stale/missing encoder.

FROM_SCRATCH_STAGES_V2 = [
    ("extract_master_dataset", "extract_master_dataset.py",
     ["--mimic4_path", "../mimic4_full", "--output_path", "../data"]),
]
CORE_STAGES_V2 = [
    ("build_irregular_cohort", "m2_build_irregular_cohort.py", []),
    ("add_t_hours", "m2_add_t_hours.py", []),
    ("static_only_mlp", "m2_static_only_mlp_v2.py", []),
    ("xgboost_tuning", "m2_xgboost_tuning_v2.py", []),          # validation-only; never opens test_v2.csv
    ("xgboost_matched", "m2_xgboost_matched_v2.py", []),        # requires xgboost_tuning's output (Phase 2a)
    # mTAN: matched-rigor tuning (3 seeds, 4 stages, noise + ECE guards) closes Deviation 15. The earlier
    # lighter-protocol m2_mtan_tuning_v2.py is retained for the record but is no longer required.
    ("mtan_tuning_matched", "m2_mtan_tuning_matched_v2.py", []),
    ("mtan_final", "m2_mtan_final_v2.py", []),                  # requires mtan_tuning_matched's config
    ("temporal_only_grud", "m2_temporal_only_grud_v2.py", []),  # writes the 5 seed checkpoints every downstream stage needs
    ("frozen_control_tuning", "m2_frozen_control_tuning_v2.py", []),   # Phase 2b; requires temporal_only_grud checkpoints
    ("frozen_temporal_control", "m2_frozen_temporal_control_v2.py", []),  # requires frozen_control_tuning's output
    ("hybrid_head_tuning", "m2_hybrid_head_tuning_v2.py", []),  # Phase 2b; requires temporal_only_grud checkpoints
    ("hybrid_grud", "m2_hybrid_grud_v2.py", []),                # primary model; requires hybrid_head_tuning's output
    ("test_mondrian_lac_validity", "test_mondrian_lac_validity.py", []),  # regression guard (Deviation 17); exit!=0 fails the run
    ("conformal", "m2_conformal_v2.py", []),                    # MAPIE + ECI + Mondrian-LAC, 3 GRU-D-family models
    ("news2_sensitivity", "m2_news2_sensitivity.py", ["--split", "chronological"]),  # test_v2.csv, same population as every reported model result
    ("recording_confound_check", "m2_v2_recording_confound_check.py", []),  # Phase 3; reads saved predictions only
    ("analyze", "m2_v2_analyze.py", []),                        # patient-cluster bootstrap, 5 deltas + Holm correction
    # Mondrian-ECI record (M2_MondrianECI_PreRegistration.md Parts 1-4): pilot -> confirmatory -> anchor correction -> comparators.
    ("mondrian_eci_pilot", "m2_mondrian_eci_pilot.py", []),
    ("mondrian_eci_confirmatory", "m2_mondrian_eci_confirmatory.py", []),      # imports the pilot's constants
    ("mondrian_eci_anchor_correction", "m2_mondrian_eci_anchor_correction.py", []),  # reads the confirmatory CSV
    ("mondrian_eci_comparators", "m2_mondrian_eci_comparators.py", []),        # pre-registered Part 4
    ("eci_vs_recalibration", "m2_eci_vs_recalibration.py", []),                # M2_ECI_vs_Recalibration_PreRegistration.md
    # Label lag (M2_LabelLag_PreRegistration.md): inference-only on saved predictions (~5 min); tests T8-T13 are hard asserts.
    ("label_lag", "m2_label_lag.py", []),
    ("logit_vs_clipping", "m2_v2_logit_vs_clipping.py", []),                    # small required ablation, 15 runs
    ("latency", "m2_v2_latency.py", []),                                       # one standardized benchmark per update path; hardware in the JSON
    # Horizon ablation (CLAUDE.md M2 required ablation): ~4h GPU; retrains encoders per horizon under M2_V2_HORIZON.
    ("horizon_ablation", "run_horizon_ablation_v2.sh", []),
    ("horizon_analyze", "m2_horizon_analyze.py", []),
    ("table_lag", "m2_generate_table_lag.py", []),               # label-lag booktabs table
    ("table_g7", "m2_generate_table_g7.py", []),                # Phase 4; clinician-facing conformal decision table
    ("tables_v2", "m2_generate_tables_v2.py", []),              # Phase 4; all booktabs tables incl. comparators + horizon
    ("report_figures", "m2_generate_report_figures_v2.py", []),
]

FROM_SCRATCH_STAGES_T2 = [
    ("extract_master_dataset", "extract_master_dataset.py",
     ["--mimic4_path", "../mimic4_full", "--output_path", "../data"]),
    ("data_general_processing", "data_general_processing.py",
     ["--master_dataset_path", "../data", "--output_path", "../data"]),
]
CORE_STAGES_T2 = [
    ("run_baseline_models", "run_baseline_models.py", []),
    ("run_news2_baseline", "run_news2_baseline.py", []),
    ("baseline_multiseed", "m2_baseline_multiseed.py", []),
    ("grud_tuning", "m2_grud_tuning.py", []),
    ("train_grud_conformal", "train_grud_conformal.py", []),
    ("ablations", "m2_ablations.py", []),
    ("eci_lr_sweep", "m2_eci_lr_sweep.py", []),
    ("multiseed_run", "m2_multiseed_run.py", []),
    ("multiseed_analyze", "m2_multiseed_analyze.py", []),
    ("static_only_mlp", "m2_static_only_mlp.py", []),
    ("xgboost_matched", "m2_xgboost_matched.py", []),
    ("temporal_only_grud", "m2_temporal_only_grud.py", []),
    ("frozen_temporal_control", "m2_frozen_temporal_control.py", []),
    ("hybrid_experiment", "m2_hybrid_experiment.py", []),
    ("hybrid_conformal", "m2_hybrid_conformal.py", []),
    ("build_predictions_all_models", "m2_build_predictions_all_models.py", []),
    ("hybrid_analyze", "m2_hybrid_analyze.py", []),
    ("hybrid_subgroups", "m2_hybrid_subgroups.py", []),
    ("mondrian_conformal", "m2_mondrian_conformal.py", []),
    ("news2_sensitivity", "m2_news2_sensitivity.py", []),
    ("generate_report", "generate_m2_report.py", []),
    ("report_figures", "m2_generate_report_figures.py", []),
]


def run_stage(name: str, script: str, extra_args: list[str]) -> None:
    script_path = os.path.join(SCRIPTS, script)
    if not os.path.exists(script_path):
        raise SystemExit(f"stage '{name}': script not found at {script_path}")
    cmd = (["bash", script] if script.endswith(".sh") else [PY, script]) + extra_args
    print(f"\n=== stage: {name} ===")
    print(f"$ {' '.join(cmd)}  (cwd={SCRIPTS})")
    result = subprocess.run(cmd, cwd=SCRIPTS)
    if result.returncode != 0:
        raise SystemExit(f"stage '{name}' failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                         help="e.g. configs/paper_track_m2.yaml (printed for a config-consistency check -- "
                              "stage scripts read their own frozen constants/artifacts, not this file)")
    parser.add_argument("--track", choices=["v2", "t2"], default=None,
                         help="overrides the config file's `track:` value")
    parser.add_argument("--from-scratch", action="store_true",
                         help="also run the raw-MIMIC-IV-ED extraction stage(s) (requires local "
                              "credentialed MIMIC-IV-ED access under ../mimic4_full)")
    parser.add_argument("--stage", default=None, help="run only this single stage by name, for debugging")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    track = args.track or cfg.get("track", "v2")
    print(f"config: {args.config}")
    print(f"  track={track}  confirmatory_seeds={cfg['seeds']['confirmatory']}  "
          f"bootstrap_n={cfg['seeds']['bootstrap_n_resamples']}")
    if track == "v2":
        print(f"  gate_v2_status={cfg['gates']['outcome_definition_gate_v2_status']}")
    else:
        print(f"  gate_v1_status={cfg['gates']['outcome_definition_gate_v1_status']}")

    if track == "v2":
        stages = (FROM_SCRATCH_STAGES_V2 if args.from_scratch else []) + CORE_STAGES_V2
    else:
        stages = (FROM_SCRATCH_STAGES_T2 if args.from_scratch else []) + CORE_STAGES_T2

    if args.stage:
        stages = [s for s in stages if s[0] == args.stage]
        if not stages:
            raise SystemExit(f"unknown stage '{args.stage}' for track '{track}'")

    for name, script, extra_args in stages:
        run_stage(name, script, extra_args)

    print(f"\nAll {track} stages completed. See data/m2_v2_results.csv "
          f"(or data/results_hybrid.csv for track=t2) and M2_Findings_Dossier.md for outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
