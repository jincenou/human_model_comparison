# Validation of the November 12 source snapshot

- Portable numerical/pair-contract tests plus auxiliary ED-P and dynamic metric tests: 29 passed. One optional LPIPS test initially skipped because its local weight configuration was not supplied.
- That LPIPS test was subsequently run in the existing metric environment with the frozen local AlexNet/linear weights: passed. It checks identical-image distance and agreement with the official loader while network downloads are forbidden.
- Public-tree validation passed: Python/JSON syntax, Markdown links, text-only bounded files, no personal server paths or credential patterns. `git diff --check` passed.
- The local gallery has100 identity pages,526 source panels and526×16 orbit images;74 missing method/person entries remain visible. Local links and a JavaScript DOM-stub interaction test passed. This is not a real-browser layout test or a full human review of every image.

These checks validate arithmetic, contracts and source packaging. They do not prove end-to-end reproduction on a clean machine, anatomical correctness, metric physical scale, or independent held-out performance. Upstream repositories, weights and permitted input data remain external dependencies.
# Reference-revision and dynamic closeout update

The updated public tree passed the release checker (312 text files before this documentation update), with no oversized binaries, personal server paths or credential-pattern matches. All 281 recorded published source/metadata hashes matched. Portable tests passed 18 checks with one optional LPIPS test skipped in that environment; dynamic and ED-P auxiliary tests passed 11 checks.

On the experiment server, the original core scores were reproduced before rescoring all 15 available pairs affected by three source-reference corrections. The frozen metric and ED-P protocols were retained. The merged cohort has 526 valid core pairs and 509 valid ED-P pairs; the original common-person populations are unchanged. Dynamic export verified 82 identities, 2,421 real exposures and 2,339 adjacent pairs per method.

The updated 100-person gallery contains 526 source pairs, 526 sets of 16 orbit views and 9,042 image files. Image hashes and 1,230 literal HTML links were checked. The revised part overlays and eight-person overview were inspected; this is not a claim that all 8,416 orbit frames received manual review or that browser layout was tested. The advisor workbook was checked by XML/cell roundtrip and its 49-file ZIP passed CRC validation. Dataset images and this local ZIP are not part of the Git release.

The opening section records earlier checks and remains historical evidence.

## Fractional-time backend expansion

Five focused tests passed in both the existing training and metric environments, covering exact endpoints, non-rounded exposures, invalid/stale brackets, eigenframe sign ambiguity and physical anisotropic covariance mixtures. All eight selected production runtimes were compared with their existing accepted source-time snapshots: maximum position discrepancy was below 1.2e-7, maximum covariance-entry discrepancy below 4.7e-10, and full SH coefficients were identical. These checks validate sampling compatibility; they do not claim that pending retrained assets have already passed visual review or scoring.
