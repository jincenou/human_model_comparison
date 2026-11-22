# Experiment workflow index

The 2026-11-22 source update includes later static-method integrations, geometry/timing, native-scene re-extraction, temporal scoring and backend ablations. Follow the [complete experiment index](EXPERIMENT_INDEX_zh-CN.md) and [file inventory](CODE_INVENTORY_20261122.md).

Paths below are relative to `workspace/human_model_comparison`.

| Task | Entry points | Required inputs |
|---|---|---|
| Core source scoring | `scripts/run_human_metric_pipeline_v1.py` | Audited references, native renders and registration certificates |
| ED-P | `scripts/run_human_edp_common.py` | Own foreground masks, source ROI, pose and feature weights |
| Dynamic metrics | `scripts/score_dynamic_source_clip.py` | Matched real exposures, reviewed references and rendered sequences |
| Added static methods | `scripts/run_human_static_two_models_staged.py` | Completed native outputs from `run_human_static_two_models_20261115.py` |
| Human geometry and timing | `evaluation_static_cd_fscore_time_v1/scripts/` | Frozen single-scan reference, scale and native-depth protocol |
| Native-scene re-extraction | `scripts/native_reaudit_*` | Original scene checkpoints and actor mapping |
| Backend ablation | `scripts/run_backend_ablation_batch_v2.py` | Frozen frontend cache and predeclared cohort |
| Orbit galleries | `scripts/common_human_orbit_batch.py` | Registered assets and native rendering environments |

Historical result bundles keep their original protocol and date. In particular the earlier source-metric core and the later aligned appearance branch are distinct implementations; no result is recomputed by this code update. Original controller/GPU-lease state must not be reused on a fresh installation.

Read [REPRODUCING.md](REPRODUCING.md) for installation, external assets, path templates and hash boundaries. Production pipelines are organized in [OmniSim_Asset](https://github.com/jincenou/OmniSim_Asset).
