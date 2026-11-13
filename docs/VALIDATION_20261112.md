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

## Scene input exposure mapping

Across 67 retained OmniRe scene directories, all 9,954 processed JPEGs matched native source-image SHA256 values along the first/next/last scene sample chain. All 99 extractable person records now have verified camera-exposure spans. The other five methods’ 500 input-budget rows are unchanged. The local advisor v3 bundle includes the updated budgets; its combined metric table and eight-person overview are byte-identical to v2. Workbook cell roundtrip and ZIP CRC checks passed. This updates input provenance only, not scores or scene training.

## November 13 ablation closeout

All eight predeclared identities completed native, optimized and matched-budget control training and source scoring. The 24 scored outputs were checked through their input/result/image hashes, shared continuation sampling and learning-rate traces, and eight per-person visual comparisons. All eight production snapshot replays match the existing source RGB and alpha exactly. No person was removed for poor quality. The new advisor bundle contains 12 verified workbook sheets and retains the unchanged six-model metric table and eight-person overview; the one-person ablation snapshot is replaced by paired eight-person, stratified and individual tables.

Final November 13 delivery checks verified all 100 reference RGB/mask/camera records, 526 source-result files, 509 valid ED-P entries, 9,042 gallery image hashes and all 54 advisor bundle files. Nineteen public numerical/pair-contract tests passed using the actual pinned LPIPS weights; 16 dynamic, ED-P and source-time sampling tests also passed (35 total, no skips). The public source release checker passed 325 text files and all 294 source-provenance entries matched.
