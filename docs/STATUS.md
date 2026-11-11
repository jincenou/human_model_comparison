# Status of this research snapshot

This document describes the source snapshot prepared on **2026-11-11 (Asia/Shanghai)**. It is not a live job monitor.

## Evidence available

- The frozen cohort contains 100 identities: retained original75 plus approved additions. The source selection SHA256 is recorded in `metadata/cohort.json`.
- AH and original TRELLIS completed native generation/reload for 100 people in the experiment workspace. Their previews still use native/estimated cameras.
- Ours has 99 accepted native reloads; HN0233 remains a coordinate-render diagnostic failure. This is not a count of registered/scored source views.
- Existing OmniRe extraction produced 99 accepted native objects; one was filtered at native initialization. SplatAD has 85 accepted objects, with missing objects audited separately.
- The reference review inspected all 100: 92 visually passed; eight body-occlusion cases remain excluded/pending. HN0321 has an additional carried-leash occlusion question. The reviewed source contracts have 91 records without pending reference flags. Final eligibility has not been frozen.
- A fixed three-person OmniRe/SplatAD source-camera diagnostic computed PSNR, SSIM, LPIPS and IoU. It is not the final paired benchmark and no leaderboard is released.
- Ours source-pose mapping was traced for HN0004, HN0035, HN0039 and HN0040. Endpoint keypoint projection agreement is below 0.003 pixels. This proves consistency with existing source-pose records, not accuracy of the estimated pose or final rendering registration.

## Work still required

1. Finish Ours/AH/TRELLIS source-camera registration and independent render validation. Certify shared camera/scale handling before expanding metric batches.
2. Finalize source-only reference eligibility, alpha conventions and the complete ED-P extractor/visibility protocol.
3. Validate/freeze shared fixed-pose orbit settings and generate the full image/coverage tables.
4. Finish ExAvatar/Jacobian fitting pilots, then FlexiAvatar and other method-specific human pilots in the planned order.
5. Complete the fixed-frontend-cache backend ablation, keeping actual optimization budgets explicit.
6. Release final paired per-person scores, input budgets, coverage/NA reasons and reproducibility manifests after acceptance.

Historical controllers and preparation state are not published as live facts. Do not interpret retained `metric_eligible=false` flags as a mistake to toggle; each requires its corresponding acceptance evidence.


## Preparation update

- TRELLIS.2 native generation/reload: 42/100 passed, 58 queued at the observed snapshot.
- The human resume controller now uses a 10 GiB data-disk reserve and respects existing vehicle GPU reservations; no human child was active during takeover.
- Jacobian SMPL neutral passed CPU forward/backward and exact conversion-equivalence checks. MHR runtime, transfer mapping and full fitting/training remain incomplete.
