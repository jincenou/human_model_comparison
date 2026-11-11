# Status of this research snapshot

This document describes the source snapshot prepared on **2026-11-11 (Asia/Shanghai)**. It is not a live job monitor.

## Evidence available

- The frozen cohort contains 100 identities: retained original75 plus approved additions. The source selection SHA256 is recorded in `metadata/cohort.json`.
- AH and original TRELLIS completed native generation/reload for 100 people in the experiment workspace. Their previews still use native/estimated cameras.
- Ours has 99 accepted native reloads; HN0233 remains a coordinate-render diagnostic failure. This is not a count of registered/scored source views.
- Existing OmniRe extraction produced 99 accepted native objects; one was filtered at native initialization. SplatAD has 85 accepted objects, with missing objects audited separately.
- Source-only reference eligibility is frozen: 91 eligible, eight body-occlusion cases and HN0321 carried-item occlusion recorded as NA. All 100 remain in the denominator.
- The incremental source-view batch computed 149 core-metric rows: OmniRe 89/100 and SplatAD 60/100, with 60 shared identities. All six prior pilot rows reproduced. A real repeat run reused all 149 results without recomputation. The full five-method benchmark and ED-P remain incomplete; no final leaderboard is released.
- Ours source-pose mapping was traced for HN0004, HN0035, HN0039 and HN0040. Endpoint keypoint projection agreement is below 0.003 pixels. This proves consistency with existing source-pose records, not accuracy of the estimated pose or final rendering registration.

## Work still required

1. Finish Ours/AH/TRELLIS source-camera registration and independent render validation. Certify shared camera/scale handling before expanding metric batches.
2. Complete and validate the ED-P extractor/visibility protocol; apply the same accepted image/alpha contract when integrating further renderers.
3. Validate/freeze shared fixed-pose orbit settings and generate the full image/coverage tables.
4. Finish ExAvatar/Jacobian fitting pilots, then FlexiAvatar and other method-specific human pilots in the planned order.
5. Complete the fixed-frontend-cache backend ablation, keeping actual optimization budgets explicit.
6. Release final paired per-person scores, input budgets, coverage/NA reasons and reproducibility manifests after acceptance.

Historical controllers and preparation state are not published as live facts. Do not interpret retained `metric_eligible=false` flags as a mistake to toggle; each requires its corresponding acceptance evidence.


## Preparation update

- TRELLIS.2 native generation/reload: 42/100 passed, 58 queued at the observed snapshot.
- The human resume controller now uses a 10 GiB data-disk reserve and respects existing vehicle GPU reservations; no human child was active during takeover.
- Jacobian SMPL neutral passed CPU forward/backward and exact conversion-equivalence checks. MHR runtime, transfer mapping and full fitting/training remain incomplete.

Incremental preparation, immutable result keys, failure reporting and CSV/HTML exports are documented in the [batch workflow](METRIC_BATCH_WORKFLOW_zh-CN.md). The user-facing combined comparison is deferred until Ours/AH/TRELLIS source registration also passes.
