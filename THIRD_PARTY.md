# Third-party and source provenance

This repository contains first-party experiment orchestration and a portable metric implementation. Its workspace adapters integrate independently distributed research projects. Their code, data and weight licenses remain applicable; this source release does not grant rights to third-party models or datasets.

- nuScenes data must be obtained from its provider under the applicable data terms.
- [Asset Harvester](https://github.com/NVIDIA/asset-harvester), [TRELLIS](https://github.com/microsoft/TRELLIS), [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), ExAvatar, JacobianAvatar, FlexiAvatar, DriveStudio/OmniRe, SplatAD and the Ours frontend/backend retain their upstream terms.
- The metric implementation uses NumPy, SciPy and Pillow. Optional LPIPS uses PyTorch, torchvision and the upstream LPIPS model/weights. See their distributions for licenses.
- SMPL, SMPL-X, FLAME, MHR and derivative body-model resources are not distributed. Public accessibility of a mirror does not replace the original terms.
- Model sources and binary CUDA extensions are not vendored in this release. Small adapters may be based on local experiment integrations; existing provenance comments and source hashes are retained.

`metadata/source_provenance.json` maps published workspace files to their original relative source paths, original SHA256 and release SHA256. Publication replaces personal installation prefixes with configurable roots; this changes code bytes. Native runtime checks must be regenerated for a fresh installation instead of assuming original manifests match the rewritten source.

**Project license:** a blanket project-wide license has not been selected in this preview. This is an explicit licensing limitation, not an assertion that all files are permissively licensed. Confirm authorship and desired terms before adding a root LICENSE or claiming OSI-approved open-source status.

