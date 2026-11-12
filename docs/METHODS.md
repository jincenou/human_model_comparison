# Method integration and dependency boundaries

The current comparison includes Ours, AH, original TRELLIS, TRELLIS.2, OmniRe and SplatAD. New integrations are deferred; StreetGS was stopped after the bounded extraction investigation. Method preparation does not establish a valid human benchmark result.

| Method | Integration | Inputs and important constraints |
| --- | --- | --- |
| Ours | 4DAnyone Turbo conditioning + full-budget Grid4D; first-party orchestration in `workspace/human` | Real sequence plus generated auxiliary views; preserve full frame/resolution/budget and SH3; non-looping source-clock query |
| Asset Harvester | Official image-directory AHC pipeline | One real source image; inferred camera, native multiview generation/lifting; registration still required |
| TRELLIS original | Native image-to-Gaussian adapter | One frozen real RGBA; original model and serializer; native preview camera is not the real source camera |
| TRELLIS.2 | Native static/PBR adapter | Method-specific representation and renderer; 97 native assets and 52 valid source-view results; failures retained |
| OmniRe | Existing DriveStudio scene/checkpoint extraction | Non-SMPL DeformableNodes variant explicitly retained; time query with native deform/root interpolation |
| SplatAD | Existing native scene/checkpoint extraction | Keep decoder and native actor/quaternion semantics; stationary-filtered objects remain unavailable |
| StreetGS | Bounded DriveStudio integration pilot | Stopped: no reliable independent person node in the inspected scenes; no full-scene retraining |
| ExAvatar | Official fitting tools and isolated environment | SMPL-X, FLAME versions/mappings and complete fitting dependencies; base-layer validation is not complete fitting |
| JacobianAvatar | Official multi-component workflow | Separate training/MHR environments, PromptHMR, WAFT, Sapiens, SAM and body-model transfer resources |
| FlexiAvatar | Subsequent integration | Full workflow and real-only variant must be recorded separately |
| Hunyuan3D-2.1 / ReconViaGen | Static human pilots | Respect actual input requirements; never duplicate photos or substitute generated inputs for required real views |

## Source pins recorded during preparation

| Source | Revision |
| --- | --- |
| [Asset Harvester](https://github.com/NVIDIA/asset-harvester) | `4e87f859f373cd468c24aca78ebfdacb45d99f33` |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| [ExAvatar](https://github.com/mks0601/ExAvatar_RELEASE) | `518f5647c8fa75cb648315c1677fdc105101fe11` |
| [JacobianAvatar](https://github.com/wcy199705/JacobianAvatar) | `ba17da4cb7c10d88b30ee501fafa870139ebd27a` |
| [FlexiAvatar](https://github.com/yihalem1/FlexiAvatar_Release) | `88f580b4d495322ffe009784057183e32691a5c2` |

For sources exported without their Git metadata, source/archive hashes in the preparation scripts and manifests are the available evidence; do not infer an upstream revision from an enclosing workspace Git repository.

## Environments and weights

`environments/` records validated core versions, not complete cross-platform lockfiles. CUDA extensions (PyTorch3D, MMCV, rasterizers, gsplat and sparse operations) must match the specific method's Python/PyTorch/CUDA stack. The installation scripts retain exact source/archive checks where implemented. Read them before execution; some old build attempts remain as provenance.

Recent preparation used Python 3.10 / torch 2.6.0+cu124 for ExAvatar and Python 3.9 / torch 2.5.1+cu121 for Jacobian training. MHR requires its separate upstream runtime. A successful import or isolated CUDA operator test does not certify an end-to-end human fit.

ModelScope discovery found community-mirror SMPL-X male/female NPZ and SMPL neutral PKL assets. Male/female NPZ files passed checksums and the ExAvatar vendored base-layer forward/backward checks. SMPL neutral required removal of a legacy Chumpy container; all model arrays and 300 shape directions were retained. Zero/nonzero pose forward outputs and gradients matched exactly in the Jacobian environment after conversion, and its consumer path is linked to the system-disk copy. This validates the SMPL component, not the full Jacobian fitting workflow. This does not establish FLAME compatibility or redistribution permission. Full FLAME 2019/2020 and companion mapping availability was not established in the inspected repositories. Use original license-compliant sources and record all versions/hashes.

Raw weight files are excluded from Git. Preparation scripts may refer to additional inventory/provenance JSON files from the private experiment directory; regenerate or provide these before running their guarded downloads. Do not bypass an expected hash or replace a missing official dependency with a similarly named model.

