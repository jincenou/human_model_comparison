"""Explicit, source-bound cycles for baked Gaussian humans.

A cycle joins two similar interior motion windows, after the ordinary asset
reader has removed its recorded root motion. The same Gaussian identities are
interpolated; two opaque avatars are never superimposed. This is motion reuse,
not a skeleton solver or a guarantee of foot locking.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from hasim.shared.assets.library_cloud import GaussianAssetFrame, interpolate_covariances, slerp
from hasim.shared.contracts.errors import AssetError

CYCLE_FORMAT = "hasim_human_cycles_v1"


def runtime_fingerprint(runtime) -> str:
    """Hash array content rather than NPZ compression or producer filenames.

    The same profile can therefore accompany an unchanged minimal export.
    Include calibration, root motion, appearance and all baked geometry.
    """
    digest = hashlib.sha256()
    for key in sorted(runtime.arrays):
        value = np.ascontiguousarray(runtime.arrays[key])
        digest.update(json.dumps([key, value.dtype.str, value.shape], separators=(",", ":")).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CycleProfile:
    asset_id: str
    start_s: float
    end_s: float
    blend_half_window_s: float
    source_duration_s: float
    source_sha256: str
    mode: str
    native_speed_mps: float | None = None
    min_playback_rate: float = .5
    max_playback_rate: float = 2.
    qualified: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        values = (self.start_s, self.end_s, self.blend_half_window_s,
                  self.source_duration_s, self.min_playback_rate, self.max_playback_rate)
        if not all(np.isfinite(x) for x in values):
            raise AssetError("Cycle times and rate bounds must be finite")
        if not self.asset_id or self.mode not in {"locomotion", "stationary"}:
            raise AssetError("Cycle requires an asset ID and locomotion/stationary mode")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256):
            raise AssetError("Cycle requires a SHA256 fingerprint of its source arrays")
        width = self.blend_half_window_s
        if width <= 0 or self.start_s < width or self.end_s + width > self.source_duration_s + 1e-12:
            raise AssetError("Cycle transition windows must lie inside the source clip")
        if self.period_s <= 2 * width:
            raise AssetError("Cycle transition must leave a non-blended motion interval")
        if not 0 < self.min_playback_rate <= self.max_playback_rate:
            raise AssetError("Invalid cycle playback rate bounds")
        if self.mode == "locomotion" and (self.native_speed_mps is None
                or not np.isfinite(self.native_speed_mps) or self.native_speed_mps <= .02):
            raise AssetError("Locomotion cycle requires a finite positive stride speed")
        if self.native_speed_mps is not None and (not np.isfinite(self.native_speed_mps) or self.native_speed_mps < 0):
            raise AssetError("Invalid cycle native speed")
        if not isinstance(self.qualified, bool) or not isinstance(self.diagnostics, dict):
            raise AssetError("Invalid cycle qualification metadata")

    @property
    def period_s(self) -> float:
        return float(self.end_s - self.start_s)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CycleProfile":
        if not isinstance(value, dict):
            raise AssetError("Cycle profile must be a JSON object")
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise AssetError(f"Invalid cycle profile: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def validate_asset(self, asset) -> None:
        if not self.qualified:
            raise AssetError("Cycle has not passed explicit qualification")
        if asset.kind != "human" or asset.id != self.asset_id:
            raise AssetError("Cycle profile belongs to a different human asset")
        if not np.isclose(asset.duration_s, self.source_duration_s, rtol=0, atol=1e-8):
            raise AssetError("Cycle source duration differs from asset metadata")
        if asset.playback_capabilities.get("heading_normalized") is not True:
            raise AssetError("Cycle requires recorded body-heading normalization")
        allowed = ({"walking", "jogging", "running"} if self.mode == "locomotion"
                   else {"standing", "gesturing", "standing_or_small_gesture"})
        if asset.motion_type not in allowed:
            raise AssetError("Cycle mode disagrees with the source motion type")


class CycleCatalog:
    """Small external qualification file; never edits the producer library."""

    def __init__(self, profiles=()):
        rows = tuple(profiles)
        self.profiles = {p.asset_id: p for p in rows}
        if len(rows) != len(self.profiles):
            raise AssetError("Duplicate cycle profiles for one asset")

    @classmethod
    def load(cls, path: str | Path) -> "CycleCatalog":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("format") != CYCLE_FORMAT or not isinstance(data.get("profiles"), list):
                raise AssetError("Expected hasim_human_cycles_v1 with a profiles list")
            return cls(CycleProfile.from_dict(row) for row in data["profiles"])
        except (OSError, ValueError, TypeError) as exc:
            if isinstance(exc, AssetError):
                raise
            raise AssetError(f"Cannot read cycle catalog {path}: {exc}") from exc

    def get(self, asset_id: str) -> CycleProfile | None:
        return self.profiles.get(asset_id)


class CyclicRuntimeAsset:
    """Stateless sampling from cumulative source seconds, including random seek.

    Around a wrap, delta in [-width, width] blends source(end+delta) into
    source(start+delta) using a quintic weight. Thus both sides of a wrap are
    evaluations of the same local function, rather than unrelated endpoints.
    Interior cut times must be between baked frames so the source linear
    interpolation itself does not introduce a derivative discontinuity there.
    """

    def __init__(self, runtime, profile: CycleProfile):
        profile.validate_asset(runtime.asset)
        if runtime_fingerprint(runtime) != profile.source_sha256:
            raise AssetError("Cycle source fingerprint does not match loaded geometry/calibration")
        times = np.asarray(runtime.arrays["timestamps"], dtype=float)
        times = times - times[0]
        for cut in (profile.start_s, profile.end_s):
            if np.min(np.abs(times - cut)) <= 1e-8:
                raise AssetError("Cycle cut must lie between source frames for a continuous seam velocity")
        self.runtime, self.profile, self.asset = runtime, profile, runtime.asset
        self.files_used = runtime.files_used

    @property
    def duration_s(self) -> float:
        return math.inf

    @property
    def period_s(self) -> float:
        return self.profile.period_s

    @property
    def native_speed_mps(self) -> float | None:
        return self.profile.native_speed_mps

    def source_samples(self, source_time_s: float) -> tuple[float, float | None, float]:
        elapsed = float(source_time_s)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise AssetError("Cycle source time must be finite and nonnegative")
        p = self.profile
        phase = elapsed % p.period_s
        width = p.blend_half_window_s
        if width <= phase <= p.period_s - width:
            return p.start_s + phase, None, 0.
        delta = phase if phase < width else phase - p.period_s
        u = np.clip((delta + width) / (2 * width), 0., 1.)
        # The quintic can round slightly above 1 near the transition endpoint.
        # Covariance interpolation deliberately rejects extrapolation.
        blend = float(np.clip(u**3 * (10 + u * (-15 + 6 * u)), 0., 1.))
        return p.end_s + delta, p.start_s + delta, blend

    def sample(self, source_time_s: float = 0.) -> GaussianAssetFrame:
        first, second, fraction = self.source_samples(source_time_s)
        a = self.runtime.sample(first)
        if second is None:
            return a
        b = self.runtime.sample(second)
        if a.means.shape != b.means.shape or not np.array_equal(a.opacities, b.opacities) or not np.array_equal(a.sh, b.sh):
            raise AssetError("Cycle requires stable Gaussian identity, opacity and SH appearance")
        q, scale = interpolate_covariances(a.quaternions, a.scales, b.quaternions, b.scales, fraction)
        qa = Rotation.from_matrix(a.local_from_appearance).as_quat()[[3, 0, 1, 2]]
        qb = Rotation.from_matrix(b.local_from_appearance).as_quat()[[3, 0, 1, 2]]
        appearance = Rotation.from_quat(slerp(qa, qb, fraction)[[1, 2, 3, 0]]).as_matrix()
        return GaussianAssetFrame(((1 - fraction) * a.means + fraction * b.means).astype(np.float32),
                                  q, scale, a.opacities, a.sh, appearance)

    def manifest(self) -> dict[str, Any]:
        return {"playback": "qualified_cycle", "period_s": self.period_s,
                "clock": "unwrapped_source_seconds", "profile": self.profile.to_dict(),
                "transition": "same_gaussian_quintic_position_and_SPD_covariance",
                "root_motion": "removed_by_source_runtime_before_blending",
                "contact_verified": False, "anti_skating_verified": False}
