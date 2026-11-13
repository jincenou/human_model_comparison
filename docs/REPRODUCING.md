# Reproduction guide

## 1. Choose the reproduction scope

There are two supported levels of access:

- **Portable core:** numerical tests, explicit pair manifests, PSNR/SSIM/IoU, and optional local-weight LPIPS. This runs independently of the experiment server.
- **Workspace integration snapshot:** method adapters and producer orchestration. These require separately installed upstream repositories, licensed data/assets, prepared source packets, and per-method runtime environments. The release does not claim a one-command end-to-end rerun of every method.

Do not execute all historical installer or launcher versions sequentially. Older versions are retained to explain preparation and interface changes; inspect their guards and follow the current method documentation.

## 2. Install and configure paths

```bash
python -m pip install -e '.[test]'
export HMC_WORKSPACE="$PWD/workspace"
export NUSCENES_ROOT=/your/data/nuScenes
export CONDA_ROOT=/your/miniconda3
export HMC_ENV_ROOT=/your/system-disk/conda_envs
export HMC_WEIGHT_ROOT=/your/system-disk/model_weights/human_comparison
export HMC_MODEL_CACHE=/your/system-disk/model_weights
```

Set `DRIVESTUDIO_ROOT`, `SPLATAD_ROOT`, and `HMC_THIRD_PARTY` for the corresponding integrations. Values are resolved at runtime, not copied from a developer's home directory. `.env.example` documents variables; it is an example, not an automatically sourced file.

Templates use `${HMC_WORKSPACE}` and related placeholders. Expand a template into a local, ignored workspace file with:

```bash
hmc-materialize metadata/templates/human_model_comparison/inputs/benchmark_human100_v3/selection.json \
  workspace/human_model_comparison/inputs/benchmark_human100_v3/selection.json
```

**Hash boundary:** path substitution changes the bytes of JSON manifests. Original hashes are provenance evidence, not valid hashes of newly materialized files. After reconstructing inputs, regenerate dependent plans and their hashes using the preparation scripts. Do not disable hash assertions to make a new installation pass. Historical worker state, PID files, GPU leases, and cached pass/fail gates must not be copied to a new machine.

## 3. Obtain data and checkpoints

Use the official nuScenes `v1.0-trainval` metadata and the exact raw sample/sweep images named by the cohort's source records. Images and annotations are not redistributed here. Install each baseline from its upstream repository and obtain its required weights under the upstream terms. Keep environments isolated; avoid upgrading a shared CUDA environment to satisfy another model.

The minimal portable runtime payloads, full scene checkpoints, frontend-generated views, and historical mask images are not part of this source release. Consequently the retained historical experiment adapters need these assets to reproduce the original run. This absence is not a successful reproducibility check; report unavailable assets explicitly.

## 4. Prepare the frozen references

Relevant scripts under `workspace/human_model_comparison/scripts/`:

| Stage | Entry points | Acceptance requirement |
| --- | --- | --- |
| Cohort/source audit | `audit_nuscenes_cohort_v2.py`, `audit_candidate_runtime_decode_v2.py` | Identity, exact raw exposure, provenance and runtime files traced |
| Reference masks | `prepare_exact_reference_masks_v2.py`, reference refinement/review scripts | Source-only visible-mask review; no neighbor-frame substitution |
| Cohort freeze | `freeze_human100_v2.py` | Explicit human review; retain original 75 and approved additions |
| Common camera | `prepare_human_source_cameras_v1.py`, `build_source_comparison_contracts_v2.py` | Official K/T, exact time and reference ROI; rendering still requires registration |

`metadata/cohort.json` is the compact public identity/time inventory. Templates retain more detailed historical provenance. A reviewed mask does not alone certify metric eligibility when external occlusion or truncation is unresolved.

## 5. Produce or reuse assets

- **Ours:** reuse the existing 4DAnyone Turbo frontend and full-budget Grid4D asset; `export_ours_frozen_time_v1.py` samples the original non-looping timeline and preserves root transforms and SH3. `render_ours_frozen_time_v3.py` checks independent reload. `prepare_ours_source_registration_pilot_v1.py` and `render_ours_source_pilot_v1.py` are historical pilots. The current batch uses the v3 registration/rendering adapters and the active metric pipeline described in CURRENT_WORKFLOW.md.
- **AH:** `prepare_ah_human_pilot_v1.py`, `run_ah_human_pilot_v1.py`, and `reload_ah_human_pilot_v1.py` use the native AHC flow. A predicted AHC source view still requires reconciliation with the calibrated real camera before comparison.
- **Original TRELLIS / TRELLIS.2:** use the corresponding `run_*_human_native_v1.py` adapters and independent reload. Their native camera previews are not registered source-camera outputs.
- **OmniRe / SplatAD:** use `extract_*_human100_v2.py`; reuse existing scene checkpoints. Original75 time-aware adapters are `render_omnire_original75_time_v1.py` and `render_splatad_original75_time_v1.py`. Keep the actual native variant and unavailable-object reason.

The shared helper files in `workspace/vehicle_model_comparison` support the human adapters; this release is not a vehicle benchmark distribution. Some helpers reference additional local plans/configurations from the original deployment. A fresh installation must generate and audit its own plans before launching workers.

## 6. Register, render, then score

Validate the camera, object transform, scale, pose time, covariance and appearance coordinates. Save image/alpha outputs and registration evidence. Only then call `hmc-score` with a pair manifest (see `tools/make_synthetic_example.py` for the exact schema).

The scorer requires matching identity/time/camera hashes and a passed registration record. It verifies bytes and schema; **it cannot independently prove that a manually supplied registration record is scientifically correct**. The rendering checks and source-only reference review remain mandatory.

Main paired tables use the intersection of valid people for the named methods and metric, with the denominator fixed to 100. Do not rank methods by means over different successful subsets. Produce a separate coverage/NA table.

## 7. Resource scheduling

Original launchers share advisory GPU locks and also inspect actual processes and reservations. Never infer availability from instantaneous zero GPU utilization. Avoid starting duplicate controllers or deleting active leases.

The requested deployment policy is **10 GiB remaining data-disk space** as a scheduling threshold; it is not a 10 GiB free-VRAM rule. Historical `human_native_queue_v1.py` retains its original 20 GiB launch condition and 11 GiB stage guard for provenance. The resume controller `human_native_queue_v2.py` implements 10 GiB launch and stage guards and passed eight isolated takeover/resource tests before deployment. It reuses the pinned plan, rejects a live previous controller, and preserves worker records. It is a resume entry point, so a fresh deployment must first prepare its own audited plan and initial state. Do not modify pinned running code. Allow output-size headroom in addition to the reserve.

## 8. Backend ablation

The required ablation fixes the same input, frontend generation cache, cameras, skeleton, masks, initialization, seed and evaluation exposure. Compare native and optimized backends; report actual optimization updates, time and memory. Do not regenerate the frontend cache or silently reduce SH degree, frame count, resolution or training budget. The current release includes a completed eight-person, three-variant ablation and per-person measured budgets; this is not a 100-person ablation. See [current workflow](CURRENT_WORKFLOW.md) for the accepted entry points.

