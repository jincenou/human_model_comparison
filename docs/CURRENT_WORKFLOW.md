# Current execution workflow

The paths below are relative to `workspace/human_model_comparison`. Configure environments and obtain data/weights as described in [REPRODUCING.md](REPRODUCING.md). Historical metadata hashes must be regenerated for locally prepared inputs; path expansion does not validate a new experiment.

| Task | Entry point | Output |
|---|---|---|
| Incremental source metrics | `scripts/run_human_metric_pipeline_v1.py --active-cohort` | Verified per-pair core scores and missingness |
| Source tables | `scripts/export_active_metric_tables.py` | Shared-person means and coverage |
| ED-P | `scripts/run_human_edp_common.py` prepare/pose/features/score stages | Frozen six-part features and auxiliary scores |
| ED-P tables | `scripts/export_human_edp_tables.py` | Common-person/common-part means |
| Full gallery | `scripts/build_full_human_comparison_gallery.py` | All100 source/orbit pages, explicit NA |
| Dynamic clips | `prepare_dynamic_source_clip.py`, `launch_dynamic_source_clip.py`, `score_dynamic_source_clip.py` | Actual exposures, reviewed masks, rendering and tOF/tLP |
| Dynamic tables | `scripts/export_dynamic_metric_tables.py` | Identity-equal pilot means |
| Fixed-cache ablation | `restore_frontend_rgba_cache.py`, `prepare_backend_ablation.py`, `launch_backend_ablation.py`, `launch_backend_budget_control.py` | Verified original cache and three budget-explicit variants |
| Ablation evaluation | `evaluate_backend_ablation.py`, `export_backend_ablation_tables.py` | Same-camera source scores and training budget table |

Run stages only after their inputs exist and are reviewed. The ED-P launcher takes an explicitly prepared run name and allowed GPU; weights remain local and pinned. Dynamic exposure/mask review is required before scoring. The ablation evaluator currently accepts exact native-frame source times (the completed HN0004 pilot); fractional-time batch expansion still requires the established covariance interpolation. Do not reuse historical controller state to bypass preparation.

The full gallery shows model names, identity, principal scores, source/orbit controls and short NA reasons. Metadata and detailed protocol remain separate. Orbit angles are relative to the source camera, not guaranteed anatomical front/back views. No per-method bbox rescaling, joint edits or score-guided alignment is introduced.

See [STATUS.md](STATUS.md) for completion counts and [numerical tables](../results/20261112/README.md). New model integrations remain deferred.
