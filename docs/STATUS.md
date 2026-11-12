# Research snapshot — 2026-11-12

This is a recorded experiment snapshot, not a live job monitor or a completed paper release.

## Current coverage

The active v3 cohort contains 100 nuScenes identities. HN0035 was replaced with HN0646 at the user's explicit direction before the current comparison; methods do not replace failed identities. Reference policy v5 retains all 100, including occlusion and umbrellas. Clothing, worn backpacks and carried items remain included.

| Method | Native assets | Source core metrics | Valid ED-P |
|---|---:|---:|---:|
| Ours | 100 | 100 | 99 |
| Asset Harvester | 100 | 100 | 100 |
| TRELLIS | 100 | 91 | 88 |
| TRELLIS.2 | 97 | 52 | 47 |
| OmniRe (existing non-SMPL variant) | 99 | 99 | 91 |
| SplatAD | 84 | 84 | 84 |

There are 526 scored method/person pairs. The five-method core comparison uses 77 common identities; adding TRELLIS.2 gives 46. The auxiliary ED-P comparison additionally intersects valid body parts per person: 70 people/329 shared parts for five methods, 40 people/199 shared parts for six. Coverage counts are not interchangeable across metrics.

TRELLIS.2 has three native generation failures and 45 outputs unavailable under the fixed source-registration protocol. These remain NA. The 52 scored outputs are retained regardless of appearance or score. Method-only means describe that successful subset; cross-method means use the same people. Limited official replays found both native thin-sheet failures and an input-mask problem; they do not establish one cause for all failures. Diagnostic mask corrections did not replace formal outputs.

OmniRe lacks one independently extractable person; SplatAD lacks 16 (ten stationary filtering cases and six unmatched independent objects). Complete scenes are not retrained to recover them.

## Additional evidence

- 526 sets of 16 common orbit views and a concise all-100-person gallery were generated locally. All images/links are checked; sample inspection and JavaScript interaction checks do not establish full visual review or browser-layout testing.
- Three paired dynamic identities have 89 actual exposures and 86 adjacent pairs per method (Ours, OmniRe, SplatAD). tOF/tLP100 are pilot results, not full-cohort dynamic rankings.
- HN0004 has a fixed-cache backend ablation: 32,999 native updates, 52,999 optimized updates, and a 52,999-update matched continuation with alpha-loss weight zero. All 1,464 frontend RGBA files match historical hashes. The matched runs share their native checkpoint and sampling/LR schedules. Only one identity has completed this ablation.
- [Numeric tables](../results/20261112/README.md) include paired means, membership and coverage. Dataset pixels and large assets are not bundled.

## Remaining work

Expand audited real-time dynamic clips and fixed-cache ablations; complete input-budget/provenance disclosure and additional orbit review. No reliable 3D human surface/joint truth or real backside reference has been established, so surface, joint/acceleration and backside image-reference metrics are not reported.

ExAvatar, JacobianAvatar, FlexiAvatar, Hunyuan3D and ReconViaGen integration is deferred while the current six methods are completed. StreetGS was stopped after the bounded existing-scene audit found no reliable independent human node. GPU 0/1 remain reserved; human jobs use shared resources on 2–7 with a 10 GiB data-disk floor.
