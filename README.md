# Human Asset Comparison

**A traceable experimental framework for reconstructing and evaluating human assets from nuScenes.**

[中文说明](README_zh-CN.md) · [Reproduction](docs/REPRODUCING.md) · [Evaluation protocol](docs/EVALUATION.md) · [Methods and dependencies](docs/METHODS.md) · [Release status](docs/STATUS.md)

> **Research preview — November 2026.** This repository releases the current experiment code, metadata templates, and independently runnable core metrics. It is not a completed five-method benchmark or a fully self-contained training distribution. Cross-method source-camera registration, ED-P, some method integrations, and the backend ablation are still in progress. Native preview images are not used as interchangeable metric inputs.

## Overview

We study human assets at the **same identity, real exposure time, and observed pose**, with two complementary outputs:

1. **Source-view reconstruction:** compare rendered assets with the real input using a common calibrated camera, white background, and reference-defined region of interest.
2. **Fixed-pose orbit inspection:** expose side/back missing surfaces, blur, floaters, and appearance errors using shared camera settings. Without real backside observations, these views do not receive image-reference scores.

The frozen cohort contains **100 nuScenes identities**: 75 retained cases and 25 additions selected through real-reference review. Method failures never trigger replacement of an identity. The comparison covers Ours (4DAnyone Turbo + Grid4D), Asset Harvester, original TRELLIS, OmniRe, and SplatAD, with additional methods under integration.

```mermaid
flowchart LR
    A[nuScenes identity and exact exposure] --> B[Reviewed visible mask and calibrated camera]
    A --> C[Method-specific reconstruction or existing scene extraction]
    C --> D[Fixed-time asset and independent reload]
    B --> E[Registration and shared camera validation]
    D --> E
    E --> F[Source metrics and coverage]
    E --> G[Fixed-pose shared orbit]
```

## Design principles

- **Preserve identity and time.** Use the frozen raw exposure; do not substitute a neighboring frame or a more favorable pose.
- **Share the viewing conditions.** Register assets in 3D, render with the same camera, and apply one reference-defined ROI. Do not independently center/scale predicted silhouettes.
- **Keep native errors visible.** Preserve predicted pose, full Gaussian covariance, spherical harmonics, and required native decoders. The reference mask is never used to erase prediction errors.
- **Report missingness.** Technical failures are NA with reasons; a valid empty prediction is scored. Main comparisons use the same valid people across methods and report `n/100`.
- **Separate reconstruction from generalization.** Source images may have been used in generation, fitting, or training. Source-view scores are not automatically held-out scores.

## Quick start: core metrics

The core metric package runs without downloading model checkpoints or loading nuScenes:

```bash
git clone https://github.com/jincenou/human_model_comparison.git
cd human_model_comparison
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q

# Synthetic contract check: demonstrates I/O and arithmetic, not benchmark performance.
python tools/make_synthetic_example.py artifacts/example
hmc-score artifacts/example/pair.json --output artifacts/example/scores.json
```

For LPIPS, install `.[metrics]` and supply both local AlexNet and LPIPS linear-weight files with their expected SHA256 hashes in the pair manifest. No checkpoint is downloaded implicitly by the scorer. See [evaluation instructions](docs/EVALUATION.md).

## Experiment code

```text
src/human_asset_benchmark/       Portable metric implementation, pair scorer, path resolver
tests/                           Numerical and provenance-contract checks
tools/                           Synthetic example and release validation
metadata/cohort.json             Frozen identity/time cohort; no dataset pixels
metadata/templates/              Path-parameterized experiment metadata
metadata/source_provenance.json  Original and published source hashes
environments/                    Observed core package versions
workspace/human_model_comparison/
  scripts/                       Reference, extraction, reload, rendering and metric adapters
  model_preparation/.../scripts/  Method-specific environment/weight preparation history
workspace/human/                 First-party producer and playback orchestration
workspace/Hasim_trajectory_assets/src/
                                 Minimal first-party Gaussian runtime modules
workspace/vehicle_model_comparison/
                                 Shared camera/resource helpers used by human adapters
```

The `workspace/` layout preserves relative imports between the experimental adapters. The included producer orchestration supports historical routes beyond this benchmark; the released benchmark cohort and protocol use **nuScenes only**. Upstream model implementations, licensed body models, generated images, checkpoints, and private run directories are not bundled.

Configure installation paths using [`.env.example`](.env.example), then follow [REPRODUCING.md](docs/REPRODUCING.md). The workspace scripts are a research integration snapshot: they require method repositories, input preparation, and runtime manifests described there. Running all scripts in filename order is not a supported workflow.

## Evaluation

| Metric | Definition in this release |
| --- | --- |
| PSNR | Float64 MSE over the entire shared ROI in sRGB `[0,1]`; exact equality is positive infinity |
| SSIM | Gaussian 11×11 window, σ=1.5, population covariance, K1=0.01, K2=0.03, RGB mean |
| LPIPS | AlexNet v0.1, fixed local weight hashes, eval/float32, RGB normalized to `[-1,1]` |
| IoU | Byte mask threshold `>127`; shared ROI and raw full-canvas variants |
| ED-P | Six-part aggregation implemented; feature extraction and visibility protocol still pending |

The reference mask includes clothes, worn backpacks, and carried items. The ROI is a square centered on the reference mask bounding box with side length `ceil(1.10 × longest side)`, padded white outside the canvas, then resized to 512×512. RGB uses bicubic interpolation; masks use nearest neighbor. See the [full protocol and limitations](docs/EVALUATION.md).

## Current release scope

| Component | Evidence available | Remaining work |
| --- | --- | --- |
| Frozen cohort | 100 identities and exposure times | Final reference eligibility/protocol freeze |
| AH / original TRELLIS | Native generation and reload batches | Shared source-camera registration |
| Ours | Fixed-time runtime and full-SH reload; source mapping pilot | Source-camera rendering/registration acceptance |
| OmniRe / SplatAD | Existing-scene extraction, physical-time and projection checks | Final common-camera certification and full scoring |
| Core metric arithmetic | Numerical tests and three-person scene-method diagnostic | Complete five-method paired evaluation |
| ExAvatar / JacobianAvatar | Isolated environments and partial official component validation | Missing assets and complete fitting/training pilots |
| Additional methods / ablation | Integration code and experiment plan | Method-specific acceptance and fixed-frontend backend ablation |

No final leaderboard, paper acceptance, or state-of-the-art claim is made by this release. The detailed [status](docs/STATUS.md) distinguishes observed results from planned work.

## Data, weights, and licensing

Obtain nuScenes and method weights through their respective providers. A community mirror's repository license does not supersede the original SMPL/SMPL-X/FLAME or checkpoint terms. No model credentials, personal server paths, dataset photos, or weights are intentionally distributed here.

See [THIRD_PARTY.md](THIRD_PARTY.md) for provenance and ownership boundaries. This preview does **not** choose a blanket open-source license on behalf of all contributors. A project-wide license and paper citation will be added when confirmed; no BibTeX entry is fabricated.

## Reproducibility and contributions

Please include the repository commit, environment versions, input/camera/weight hashes, and affected identity when reporting a problem. Do not attach restricted dataset images or credentials to public issues. Changes to camera geometry, alpha handling, or missing-data aggregation should include a focused numerical test. See [CONTRIBUTING.md](CONTRIBUTING.md). An optional [GitHub Actions template](docs/ci/README.md) is included; hosted CI is not enabled by this release.

