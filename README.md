# Human Asset Comparison

Source code for human-asset preparation, method integration, source-camera registration, static and temporal metrics, observed geometry, timing, backend ablations, and qualitative inspection.

**The 2026-11-22 update organizes the experiment code. It does not rerun models or replace previously published results.**

[中文说明](README_zh-CN.md) · [Experiment index](docs/EXPERIMENT_INDEX_zh-CN.md) · [Complete code inventory](docs/CODE_INVENTORY_20261122.md) · [Reproduction](docs/REPRODUCING.md) · [Evaluation](docs/EVALUATION.md)

## Repository roles

- [OmniSim_Asset](https://github.com/jincenou/OmniSim_Asset): rigid/human production pipelines, normalization, export and reuse.
- This repository: human comparison experiments and metric/ablation implementations.
- [vehicle_model_comparison](https://github.com/jincenou/vehicle_model_comparison): rigid experiments and shared camera/rendering/geometry helpers.

## Layout

```text
src/human_asset_benchmark/       Portable metrics, explicit pair scorer, path resolver
tests/                          Numerical and provenance-contract checks
tools/                          Synthetic example and source-release validation
metadata/templates/             Parameterized protocol and cohort templates
metadata/source_snapshot_20261122.json
                                Original and released hashes for this code update
workspace/human_model_comparison/
  scripts/                      Reference, method, registration, rendering, scoring and ablation code
  evaluation_static_cd_fscore_time_v1/scripts/
                                Observed-geometry and timing experiment
  model_preparation/.../scripts/ Environment integration and feasibility history
  reports/                      Selected source-only diagnostic adapters
results/                        Existing dated public results, retained unchanged
```

The workspace preserves relative imports and historical entry points. A script's presence is not evidence that its required data, checkpoints, calibration, or reviewed inputs are available. Use the [experiment index](docs/EXPERIMENT_INDEX_zh-CN.md), not filename ordering, to select a workflow.

## Quick start: portable metric core

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python tools/check_release.py
python tools/make_synthetic_example.py artifacts/example
hmc-score artifacts/example/pair.json --output artifacts/example/scores.json
```

This synthetic example verifies the interface and arithmetic, not reconstruction performance. LPIPS requires explicit local weight files and hashes; it does not download checkpoints implicitly. The model adapters additionally require isolated upstream environments, data, prepared references and native assets.

## Experiment families

| Family | Entry points under `workspace/human_model_comparison` |
|---|---|
| Source-view RGB and silhouette | `scripts/run_human_metric_pipeline_v1.py`, `scripts/human_source_metrics_v1.py` |
| Six-part ED-P | `scripts/run_human_edp_common.py`, `scripts/human_edp_common.py` |
| Actual-exposure tOF / tLP | `scripts/prepare_dynamic_source_clip.py`, `scripts/score_dynamic_source_clip.py` |
| Hunyuan3D-2.1 / ReconViaGen | `scripts/run_human_static_two_models_20261115.py`, `scripts/run_human_static_two_models_staged.py` |
| Geometry and timing | `evaluation_static_cd_fscore_time_v1/scripts/` |
| Native scene re-extraction | `scripts/native_reaudit_*`, `scripts/score_native_reaudit_*` |
| Fixed-cache backend ablation | `scripts/prepare_backend_ablation.py`, `scripts/run_backend_ablation_batch_v2.py` |
| Common orbit inspection | `scripts/common_human_orbit_batch.py`, `scripts/render_common_human_orbit.py` |

Historical result bundles retain their original dates and protocols: [2026-11-12](results/20261112/README.md) and [backend ablation](results/20261113/ablation/RESULTS.md). Later alignment, added-method and native-extraction branches have distinct contracts. Combining their source code does not justify merging scores or changing metric denominators. See [current workflow](docs/CURRENT_WORKFLOW.md).

## Paths, dependencies and provenance

Use [`.env.example`](.env.example) and [REPRODUCING.md](docs/REPRODUCING.md). Source path literals are resolved through the existing `human_asset_benchmark.paths` mechanism. Materializing a template changes its bytes: regenerate local input and plan hashes rather than treating historical hashes as newly verified evidence.

Dataset pixels, generated assets, weights, licensed body models, process state and GPU leases are not distributed. First-party producer code and historical shared helpers already present in `workspace/` remain available; use OmniSim_Asset for the organized production pipeline and the vehicle repository for newer shared experiment helpers. Dependencies and ownership boundaries are described in [THIRD_PARTY.md](THIRD_PARTY.md).

The source release is checked with CPU numerical/contract tests and syntax/configuration validation. This update does not claim that every GPU workflow was rerun on a fresh installation.
