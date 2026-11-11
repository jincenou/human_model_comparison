# Evaluation protocol and validity

## Shared source view

Each record binds identity → scene/instance → raw RGB and microsecond timestamp → mask and K/T → model time/root motion → asset → independent render. The source camera uses the official raw-canvas calibration. Asset-specific coordinate systems must be mapped into the camera/world frame with their covariance and appearance directions preserved.

Unknown global scale and orientation require an explicit, recorded registration protocol. No per-method image-bbox normalization, best-angle search using evaluation scores, pose correction, exposure matching, or replacement of source poses is allowed. Model shape errors may still produce different silhouettes after correct common-camera rendering; forcing all predictions to exactly the same pixel height would erase part of the error being measured.

## Images and alpha

Reference: binary visible mask (`byte >127`), original sRGB inside, white outside. Preserve clothes, worn backpacks, handheld bags, bottles and phones. Reference-only review records external occlusion and truncation separately from self-occlusion.

Prediction: use its own alpha. Specify straight RGB versus already white-composited RGB. Never multiply the prediction by the reference mask. The portable CLI accepts byte RGB with `straight_uint8` or `white_composited_uint8`; the underlying metric module also supports explicitly supplied float straight/premultiplied arrays.

ROI: half-open reference bounding box; square side `ceil(1.10 * max(width,height))`; left/top are floored from the centered square. Pad RGB white and masks zero beyond the raw canvas. Resize to 512×512 with PIL bicubic RGB and nearest masks. Record the original person pixel height: resizing cannot create real detail.

## Metrics

PSNR is `-10 log10(MSE)` with float64 MSE on sRGB `[0,1]` over the full common RGB ROI. Exact matches use `status=positive_infinity`, JSON `value=null`; this is distinct from `status=na`.

SSIM uses an 11×11 separable Gaussian (σ=1.5), population covariance, dynamic range 1, K1=0.01, K2=0.03 and the RGB mean. Spatial averaging excludes the five-pixel boundary to match the validated independent implementation.

LPIPS uses AlexNet v0.1, float32/eval, NCHW RGB in `[-1,1]`. Both backbone and linear weights must match their recorded SHA256. The loader is offline and must not silently fetch a replacement.

IoU uses byte threshold `>127`; report both common-ROI IoU and raw full-canvas IoU. An empty eligible prediction has IoU zero. An empty reference is NA. A technical render failure is not replaced by an all-white image.

ED-P reserves head, torso, left/right arm, and left/right leg. Only the empty/visible-part aggregation is implemented. Keypoint selection, DINOv3 feature layer/preprocessing, and per-part visibility still require a validated frozen extractor. Current outputs explicitly report ED-P as NA, not zero.

## Pair manifest

`tools/make_synthetic_example.py` writes a fully specified example containing:

- reference and prediction paths plus SHA256 values;
- identity, exposure timestamp, and shared camera SHA256;
- a camera JSON with `identity`, `timestamp_us`, `render_wh`, `roi_xyxy`, K and T;
- a registration JSON bound by SHA256 and matching identity/time/camera;
- explicit prediction alpha encoding and reference eligibility;
- optional LPIPS configuration with `backbone`, `linear`, both expected hashes, and device.

Run from the repository root. The example uses absolute paths derived at creation time for unambiguous local file resolution. It is synthetic and is never included in cohort statistics.

## Aggregation and interpretation

Keep every frozen identity in the denominator, even if a method cannot produce an independent object. Report each method's coverage and all NA reasons. For any cross-method mean, choose the common valid identity set for that specific metric first. Include its IDs and `n/100`; do not average different sets and call it paired comparison.

Current scene-method pilot scores are diagnostic. Passing camera projection algebra does not prove final registration, source-reference eligibility, or complete evaluation validity. Generated/native preview cameras must never be mixed with real-camera scores.

Source reconstruction evaluates an observed state. Disclose all real frames, video duration, interpolated frames, generated auxiliaries, poses, cameras and other priors consumed by a method. A true held-out experiment needs a separate input audit. Backside PSNR/LPIPS requires real backside reference. Fixed-pose orbits without such observations are qualitative.
