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
- 82 paired dynamic identities have 2,421 actual exposures and 2,339 adjacent pairs per method (Ours, OmniRe, SplatAD). HN0252 (paper/forearm) and HN0321 (leash) remain reference-unavailable after bounded repair, stopped at the user's request; these are not model-generation failures. Sixteen other identities lack a common independent scene object. The denominator remains 100.
- HN0116's backpack, HN0632's foreground signage and HN0838's background chain were corrected using real source pixels. All 15 available method/person pairs affected by these three revisions were rescored for core metrics and ED-P. Camera K/T, assets and registration remain unchanged; the revised reference determines one common ROI for every method. Historical snapshots are retained locally.
- HN0004 has a fixed-cache backend ablation: 32,999 native updates, 52,999 optimized updates, and a 52,999-update matched continuation with alpha-loss weight zero. All 1,464 frontend RGBA files match historical hashes. The matched runs share their native checkpoint and sampling/LR schedules. Only one identity has completed this ablation.
- [Numeric tables](../results/20261112/README.md) include paired means, membership and coverage. Dataset pixels and large assets are not bundled.

## Remaining work

The current dynamic batch is closed with explicit missingness. Expand fixed-cache ablations and additional orbit review separately. [Input budgets](INPUT_BUDGETS.md) now cover 600 person/method rows, with exact scene-time mapping limitations disclosed. No reliable 3D human surface/joint truth or real backside reference has been established, so surface, joint/acceleration and backside image-reference metrics are not reported.

ExAvatar, JacobianAvatar, FlexiAvatar, Hunyuan3D and ReconViaGen integration is deferred while the current six methods are completed. StreetGS was stopped after the bounded existing-scene audit found no reliable independent human node. GPU 0/1 remain reserved; human jobs use shared resources on 2–7 with a 10 GiB data-disk floor.

## Backend expansion queued after the reporting closeout

An eight-person cohort is now fixed by taking the first four IDs from each original/new stratum. HN0004 remains the only completed three-variant result at this update; the seven additional identities are queued on GPUs 2–7. Original RGBA caches must match every historical file hash before training; generated videos are not regenerated. Fractional source-time sampling has passed five focused numerical checks in both training and evaluation environments, plus comparisons with eight existing production snapshots. Pending training does not add numbers to the published table.
