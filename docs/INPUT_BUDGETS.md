# Inputs and information budgets

The comparison uses the same identities and evaluation cameras, but **does not use equal reconstruction input budgets**. Counts below come from retained manifests and native scene configurations, not from model names or paper defaults. [Per-person/method CSV](../results/20261112/input_budget/per_identity_method.csv) and [hashed source audit](../results/20261112/input_budget/audit.json) contain 600 rows for the frozen 100 identities.

| Method | Real observations | Generated or resampled inputs | Additional information |
|---|---|---|---|
| Ours | 28–31 distinct exposures per person; 2.4–2.500006 seconds, one camera | 61 conditioning slots, 58–60 marked interpolated; 24 generated views × 61 frames at 352×640 | Estimated body pose/skeleton, conditioning cameras and root motion; Grid4D trains on 21 generated views and reserves 3 synthetic views |
| Asset Harvester | One frozen real source image | All 100 outputs retain 16 generated multiview images | AHC-predicted camera and pretrained generation/lifting priors |
| TRELLIS | One frozen foreground image | No externally supplied generated view set | Pretrained image-to-3D prior; source body keypoints used for rigid registration |
| TRELLIS.2 | One frozen foreground image | No externally supplied generated view set | Pretrained image-to-3D prior; source body keypoints used for rigid registration |
| OmniRe | Six-camera scene inputs: 132–240 processed images over 22–40 retained time indices for the 99 extractable identities | Existing scene checkpoint reused; no scene retraining | LiDAR, calibrated cameras, masks and native object trajectories; non-SMPL variant |
| SplatAD | Native train split: 744–1,375 images, 208–385 LiDAR scans; image span 18.692579–19.942606 seconds for the 84 extractable identities | Existing scene checkpoint reused; no scene retraining | Calibrated multi-camera/LiDAR observations, native actor tracks and decoder |

For Ours, 61 conditioning slots are not 61 independent real photographs. Interpolation does not add real observations. Likewise, reserved generated views are not real held-out ground truth. The captured native interval can exceed the generated sequence's 2.4-second support; dynamic evaluation uses the shared supported real exposures without looping or retiming.

Scene counts are shared by people from the same checkpoint and must not be added as independent per-person observations. They do not mean the target appears in every image. SplatAD counts follow the configured sensor selection and hashed native train split; training-shadow evaluation entries are not counted again. For OmniRe, the retained image/time-index counts are verified, but most processed directories lack a verified per-index physical-time mapping in this audit. Exact duration is therefore left blank where it cannot be established; two directly matched entries include their source timestamp span.

The audit establishes configured/retained inputs, not a fresh replay of every historical optimizer update. Body pose and root estimates are priors, not motion ground truth. Evaluation-mask corrections do not retroactively replace generation inputs. Source images also participate in some alignment stages, so the table describes observed-state reconstruction and must not be labeled an equal-input or strict held-out generalization benchmark.
