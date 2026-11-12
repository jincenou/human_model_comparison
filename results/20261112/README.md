# Recorded numerical results — 2026-11-12

These CSV files are an experimental snapshot, not a final leaderboard. The denominator remains 100; compare methods only within the same named pairing.

- Source: [paired means](source/paired_means.csv), [members](source/paired_members.csv), [coverage](source/coverage.csv), [missing reasons](source/missing.csv), [TRELLIS.2 own valid subset](source/trellis2_valid_means.csv).
- ED-P_common: [paired means](edp/paired_means.csv), [common people/parts](edp/paired_members.csv), [coverage](edp/coverage.csv), [per-object](edp/per_object.csv), [per-part](edp/per_part.csv), [excluded pairings](edp/paired_excluded.csv).
- Dynamic three-person pilot: [means](dynamic/paired_means.csv), [per-person](dynamic/per_identity.csv).
- Fixed-cache one-person backend ablation: [scores and measured training budgets](ablation/metrics.csv).

Source-core pairings use common77 (five methods) or common46 (six). ED-P also intersects parts, yielding common70 or common40 respectively. Own-subset TRELLIS.2 means cannot rank it against other methods evaluated on different people. Three generation failures and45 registration-unavailable outputs remain NA; no failed identities were replaced or assigned zero scores. Successful outputs are not filtered by visual quality.

Ablation native32,999 vs optimized52,999 includes20,000 extra updates. The52,999 matched continuation isolates the alpha-loss coefficient in the same staged schedule. One-person differences are not cohort-level evidence. Dynamic scores include white-background regions and estimated camera/image motion, not 3D joint truth.

Original/released file hashes are recorded in `metadata/source_provenance.json` at the repository root. Full images, checkpoints and source data must be reconstructed locally under their upstream licenses.
