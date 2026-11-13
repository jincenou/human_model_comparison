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
| Dynamic tables | `scripts/export_dynamic_metric_tables.py` | Identity-equal means and explicit reference/object missingness |
| Input-budget audit | `scripts/audit_human_input_budgets.py` | 600 person/method records, source hashes and input limitations |
| Reference revision | `rescore_reviewed_static_references.py`, `prepare_revised_reference_edp.py` | Same-asset rescoring of every affected available method |
| Advisor material | `export_advisor_figures.py`, `export_advisor_tables.py` | Eight deterministic examples, combined table, CSV/XLSX and ZIP |
| Fixed-cache ablation | `restore_frontend_rgba_cache.py`, `prepare_backend_ablation.py`, `launch_backend_ablation.py`, `launch_backend_budget_control.py` | Verified original cache and three budget-explicit variants |
| Ablation evaluation | `evaluate_backend_ablation.py`, `export_backend_ablation_tables.py` | Same-camera source scores and training budget table |

Run stages only after their inputs exist and are reviewed. The ED-P launcher takes an explicitly prepared run name and allowed GPU; weights remain local and pinned. Dynamic exposure/mask review is required before scoring. The ablation evaluator now uses the production 25 Hz frame bracket and existing SPD covariance interpolation at fractional source times. Five focused numerical checks and eight real runtime/snapshot comparisons passed; it does not round exposure times or query the deformation network using a different sampling rule. Do not reuse historical controller state to bypass preparation.

The full gallery shows model names, identity, principal scores, source/orbit controls and short NA reasons. Metadata and detailed protocol remain separate. Orbit angles are relative to the source camera, not guaranteed anatomical front/back views. No per-method bbox rescaling, joint edits or score-guided alignment is introduced.

See [STATUS.md](STATUS.md) for completion counts and [numerical tables](../results/20261112/README.md). New model integrations remain deferred.

Tables and galleries resolve `inputs/active_results.json` through `active_human_results.py`. The metadata template records the current experiment's reference revisions; private image/score artifacts must exist locally before using those paths. The original incremental scorer preserves its original qualification/cache. A newly generated result must be explicitly merged against the current reviewed reference revision before entering the updated reporting bundle; changing a results pointer alone does not recompute metrics.

Dynamic scoring only discovers reviewed references. The two terminal reference-unavailable cases have no qualification file and are not silently retried or treated as successful white images.

The completed eight-person ablation cohort was fixed before additional training: the first four lexicographic IDs from each original/new stratum, including the completed HN0004 pilot. `configs/backend_ablation_batch_v2.json` records the IDs and selection rule. `run_backend_ablation_batch_v2.py` observes existing cache-restoration handles, then chains hash-verified cache preparation, the three original-budget variants and scoring. It stops on failure without automatic retry. `export_backend_ablation_batch_v2.py` reports paired coverage against eight; all eight people have now completed all three variants and visual review. This bounded ablation is not a 100-person result.

For the current input-budget revision, `audit_omnire_input_exposures.py` reconstructs the actual native scene sample chain and verifies each processed JPEG against its source exposure. Then run `audit_human_input_budgets.py --output reports/input_budget_audit_v2 --omnire-exposures reports/omnire_input_exposure_audit_v1/manifest.json`. The `input_budget` entry in `inputs/active_results.json` selects the reviewed audit for advisor exports. Preserve earlier audits: the exposure manifest binds its source inventory hash.
