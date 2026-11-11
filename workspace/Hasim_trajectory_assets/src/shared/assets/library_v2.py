"""Read the current rigid/ and human/ delivery without legacy motion guesses.

Catalog construction reads metadata only. ``load_asset`` opens one selected
payload; no absolute producer-code path or training dependency is required.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hasim.shared.assets.library_cloud import GaussianAssetFrame, interpolate_covariances, similarity, unit_quaternions
from hasim.shared.contracts.errors import AssetError, CatalogError

VEHICLE_CATEGORIES = frozenset({"car", "truck", "van", "bus", "vehicle"})
_MINIMAL_FORMAT = "hasim_asset_runtime_v1"


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CatalogError(f"Cannot read asset metadata {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"Expected JSON object in {path}")
    return data


@dataclass(frozen=True)
class LibraryAsset:
    id: str
    category: str
    kind: str
    path: Path
    meta: dict[str, Any]
    source_files: tuple[Path, ...] = ()

    @property
    def duration_s(self) -> float:
        if self.kind != "human":
            return math.inf
        return float(self.meta["duration_s"])

    @property
    def native_speed_mps(self) -> float | None:
        value = self.meta.get("native_speed_mps")
        return None if value is None else float(value)

    @property
    def motion_type(self) -> str:
        return str(self.meta.get("motion_type", "static"))

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.meta["dimensions"])

    @property
    def playback_capabilities(self) -> dict[str, Any]:
        """Small runtime contract, preserved when producer metadata is removed.

        Heading normalization is a capability, not a naturalness certificate.
        Some source clips have an all-zero root_yaw because no Gaussian body
        heading could be recovered; those clips retain their original turn.
        """
        if self.kind != "human":
            return {}
        saved = self.meta.get("playback_capabilities")
        if saved is not None:
            return dict(saved)
        practical = self.meta.get("practical_playback", {})
        return {
            "heading_normalized": practical.get("body_heading_from_gaussians_available") is True,
            "contact_verified": practical.get("contact_verified") is True,
            "anti_skating_verified": practical.get("anti_skating_verified") is True,
            "action_verified": practical.get("action", {}).get("human_verified") is True,
            "speed_basis": practical.get("speed_basis", "unspecified"),
            "ground_basis": practical.get("ground_basis", "unspecified"),
        }


class AssetLibrary:
    """Metadata-only index of usable candidates, sorted for seeded selection.

    ``quality_approved`` is reported, not fabricated: current human assets are
    engineering candidates. Explicit skip/severe-failure entries are excluded.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.excluded: dict[str, str] = {}
        rows: list[LibraryAsset] = []
        minimal = self.root / "library.json"
        rigid = self.root / "rigid/asset_library_rigid.json"
        human = self.root / "human/drivelab_asset.json"
        release = self.root / "drivelab_asset.json"
        published_rigid = self.root / "releases/rigid/drivelab_asset.json"
        if minimal.is_file():
            rows.extend(self._minimal(minimal))
        elif release.is_file() and _json(release).get("manifest_type") == "drivelab_asset":
            rows.extend(self._release(release))
            # human_compaction_shared_index_v1
            human_runtime = self.root / "human/library.json"
            if human_runtime.is_file():
                rows.extend(self._minimal(human_runtime))
            elif human.is_file():
                rows.extend(self._human(human))
        else:
            if not rigid.is_file() and (self.root / "asset_library_rigid.json").is_file():
                rigid = self.root / "asset_library_rigid.json"
            if not human.is_file() and (self.root / "drivelab_asset.json").is_file():
                human = self.root / "drivelab_asset.json"
            # The published manifest is authoritative. Never merge stale
            # production/legacy candidates or silently fall back on corruption.
            if published_rigid.is_file():
                rows.extend(self._release(published_rigid))
            elif rigid.is_file():
                rows.extend(self._rigid(rigid))
            # human_compaction_shared_index_v1
            human_runtime = self.root / "human/library.json"
            if human_runtime.is_file():
                rows.extend(self._minimal(human_runtime))
            elif human.is_file():
                rows.extend(self._human(human))
        self.assets = tuple(sorted(rows, key=lambda item: item.id))
        self._by_id = {asset.id: asset for asset in self.assets}
        if len(self._by_id) != len(self.assets):
            raise CatalogError("Duplicate asset IDs in library")
        if not self.assets:
            raise CatalogError(f"No usable asset entries under {self.root}")

    def get(self, asset_id: str) -> LibraryAsset:
        try:
            return self._by_id[asset_id]
        except KeyError as exc:
            raise CatalogError(f"Unknown asset ID {asset_id!r}") from exc

    def candidates(self, object_type: str) -> tuple[LibraryAsset, ...]:
        kind = str(object_type).lower()
        if kind in ("human", "pedestrian", "person"):
            return tuple(a for a in self.assets if a.kind == "human")
        if kind == "vehicle":
            return tuple(a for a in self.assets if a.category in VEHICLE_CATEGORIES)
        if kind == "rigid":
            return tuple(a for a in self.assets if a.kind == "rigid")
        return tuple(a for a in self.assets if a.category == kind)

    def _release(self, index: Path):
        from hasim.shared.assets.drivelab_release import rigid_records
        for identity, category, path, meta in rigid_records(index):
            yield self._record(identity, category, "rigid", path, meta, (index,))

    def _rigid(self, index: Path):
        for entry in _json(index).get("assets", []):
            identity = str(entry.get("global_uid") or entry.get("uid") or "")
            if entry.get("status", "keep") != "keep":
                self.excluded[identity] = "status is not keep"
                continue
            path = (index.parent / entry["source_path"]).resolve()
            binding_path = path.parent / "binding.json"
            binding = _json(binding_path)
            coordinate = binding.get("coordinate", {})
            if coordinate.get("up") != "+Z" or not coordinate.get("canonical_transform_already_applied"):
                raise CatalogError(f"Expected canonical Z-up delivered PLY: {path}")
            scale = float(binding["scale"])
            if not np.isfinite(scale) or scale <= 0:
                raise CatalogError(f"Invalid metric scale for {identity}")
            if coordinate.get("metric_scale_baked", False):
                scale = 1.0
            meta = dict(entry, format="rigid_ply", binding=binding,
                        metric_scale=scale, motion_type="static",
                        dimensions=binding["size_lwh_m"])
            yield self._record(identity, str(entry["category"]), "rigid", path, meta, (index, binding_path))

    def _human(self, index: Path):
        for entry in _json(index).get("assets", []):
            identity = str(entry["global_uid"])
            if entry.get("skip_insertion") or entry.get("quality_state") == "severe_failure":
                self.excluded[identity] = str(entry.get("insertion_exclusion_reason") or entry.get("quality_state"))
                continue
            if entry.get("runtime_metadata"):
                metadata_path = (index.parent / entry["runtime_metadata"]).resolve()
                meta = _json(metadata_path)
                if meta.get("format") != _MINIMAL_FORMAT:
                    raise CatalogError(f"Unsupported migrated human format: {metadata_path}")
                if meta.get("skip_insertion") or meta.get("quality_state") == "severe_failure":
                    self.excluded[identity] = str(meta.get("insertion_exclusion_reason") or meta.get("quality_state"))
                    continue
                yield self._record(identity, "human", "human", metadata_path.parent / meta["payload"],
                                   meta, (index, metadata_path))
                continue
            package = (index.parent / entry["package_path"]).resolve()
            insertion_path = package / "insertion.json"
            insertion = _json(insertion_path)
            if insertion.get("skip_insertion"):
                self.excluded[identity] = str(insertion.get("insertion_exclusion_reason", "skip_insertion"))
                continue
            practical = insertion.get("practical_playback", {})
            if not practical.get("approximate_placement_available"):
                self.excluded[identity] = "missing practical playback calibration"
                continue
            if practical.get("coordinate_convention") != "right_handed_x_forward_y_left_z_up":
                raise CatalogError(f"Unknown human coordinate convention for {identity}")
            manifest_path = package / "manifest.json"
            manifest = _json(manifest_path)
            fps = float(manifest["fps"])
            count = int(manifest["frame_count"])
            if fps <= 0 or not np.isfinite(fps) or count < 2:
                raise CatalogError(f"Invalid human clock for {identity}")
            size = np.asarray(entry.get("size_xyz", [0, 0, 0]), dtype=float)
            size *= float(entry.get("asset_unit_to_model_meter", 1))
            meta = dict(entry, format="human_npz", duration_s=(count - 1) / fps,
                        fps=fps, frame_count=count, dimensions=size.tolist(),
                        native_speed_mps=practical.get("native_speed_model_mps"),
                        motion_type=practical.get("action", {}).get("label", "unknown"),
                        playback_path=str(package / practical.get("arrays", "placement_playback.npz")),
                        practical_playback=practical)
            path = package / manifest.get("asset", {}).get("path", "asset_4dgs.npz")
            yield self._record(identity, "human", "human", path, meta, (index, insertion_path, manifest_path))

    def _minimal(self, index: Path):
        data = _json(index)
        if data.get("format") != _MINIMAL_FORMAT:
            raise CatalogError(f"Unsupported minimal library format in {index}")
        # human_compaction_exclusions_v1
        for excluded in data.get("excluded_assets", []):
            self.excluded[str(excluded["id"])] = str(excluded.get("reason", "excluded"))
        for row in data.get("assets", []):
            metadata_path = index.parent / row["metadata"]
            meta = _json(metadata_path)
            if meta.get("format") != _MINIMAL_FORMAT:
                raise CatalogError(f"Unsupported runtime asset format in {metadata_path}")
            path = metadata_path.parent / meta["payload"]
            yield self._record(meta["id"], meta["category"], meta["kind"], path, meta, (index, metadata_path))

    @staticmethod
    def _record(identity, category, kind, path, meta, source_files):
        if not identity or not path.is_file():
            raise CatalogError(f"Asset identity or payload missing: {identity!r}, {path}")
        dimensions = np.asarray(meta["dimensions"], dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise CatalogError(f"Invalid asset dimensions for {identity}")
        return LibraryAsset(identity, category, kind, path.resolve(), meta, tuple(source_files))


class RuntimeAsset:
    """One selected asset resident in CPU memory, independent of any renderer."""

    def __init__(self, asset: LibraryAsset, arrays: dict[str, np.ndarray], files_used):
        self.asset = asset
        self.files_used = tuple(dict.fromkeys(Path(p).resolve() for p in files_used))
        self.arrays = arrays
        self._validate()

    @property
    def duration_s(self) -> float:
        return float(self.arrays["timestamps"][-1] - self.arrays["timestamps"][0]) if self.asset.kind == "human" else math.inf

    def _validate(self):
        a = self.arrays
        times = np.asarray(a["timestamps"], dtype=np.float64)
        xyz = a["frame_xyz"]
        if times.ndim != 1 or len(times) == 0 or np.any(np.diff(times) <= 0):
            raise AssetError("Asset timestamps must be strictly increasing")
        if xyz.ndim != 3 or xyz.shape[0] != len(times) or xyz.shape[2] != 3 or xyz.shape[1] == 0:
            raise AssetError("Expected frame_xyz shape F x N x 3")
        f, n, _ = xyz.shape
        shapes = {"frame_quaternions": (f, n, 4), "frame_scales": (f, n, 3),
                  "opacities": (n,), "root_xy": (f, 2), "root_yaw": (f,)}
        for key, shape in shapes.items():
            if a[key].shape != shape:
                raise AssetError(f"Invalid {key}: {a[key].shape}; expected {shape}")
        for key, value in a.items():
            if not np.isfinite(value).all():
                raise AssetError(f"Non-finite {key} in {self.asset.id}")
        if np.any(a["frame_scales"] <= 0) or np.any((a["opacities"] < 0) | (a["opacities"] > 1)):
            raise AssetError("Scales must be activated positive values, opacities in [0,1]")
        qnorm = np.linalg.norm(a["frame_quaternions"], axis=-1)
        if not np.allclose(qnorm, 1, atol=2e-4):
            raise AssetError("Expected normalized exported frame quaternions; native raw quaternions need covariance conversion")
        sh = a["sh"]
        if sh.ndim != 3 or sh.shape[0] != n or sh.shape[2] != 3 or sh.shape[1] not in (1, 4, 9, 16):
            raise AssetError("Expected complete real SH degree 0..3, N x K x 3")
        similarity(a["asset_to_standard"])
        if np.asarray(a["ground_offset"]).shape != ():
            raise AssetError("Expected scalar ground_offset")
        if self.asset.kind == "human" and not np.isclose(self.duration_s, self.asset.duration_s, atol=1e-6):
            raise AssetError("Payload duration disagrees with catalog")
        a["root_yaw"] = np.unwrap(a["root_yaw"])

    def sample(self, source_time_s: float = 0.0) -> GaussianAssetFrame:
        elapsed = float(source_time_s)
        tolerance = 8 * np.finfo(float).eps * max(1., self.duration_s) if self.asset.kind == "human" else 0.
        if not np.isfinite(elapsed) or elapsed < -tolerance:
            raise AssetError("Source time must be finite and nonnegative")
        if self.asset.kind == "human" and elapsed > self.duration_s + tolerance:
            raise AssetError(f"{self.asset.id}: source time {elapsed:g}s exceeds one-clip duration {self.duration_s:g}s; no implicit loop/hold")
        if self.asset.kind == "human":
            # Allow only arithmetic roundoff when a mathematically exact clip
            # endpoint is reconstructed from distance/speed. This is not hold.
            elapsed = min(max(elapsed, 0.), self.duration_s)
        a = self.arrays
        times = a["timestamps"]
        query = times[0] + (elapsed if self.asset.kind == "human" else 0)
        hi = min(int(np.searchsorted(times, query, side="right")), len(times) - 1)
        lo = max(0, hi - 1)
        fraction = 0.0 if hi == lo else float((query - times[lo]) / (times[hi] - times[lo]))
        def interp(key):
            return (1 - fraction) * a[key][lo] + fraction * a[key][hi]
        quaternions, scales = interpolate_covariances(
            a["frame_quaternions"][lo], a["frame_scales"][lo],
            a["frame_quaternions"][hi], a["frame_scales"][hi], fraction)
        cloud = GaussianAssetFrame(interp("frame_xyz"), quaternions, scales,
                                   a["opacities"], a["sh"], np.eye(3))
        root = interp("root_xy")
        heading = float(interp("root_yaw"))
        c, s = np.cos(-heading), np.sin(-heading)
        inplace = np.eye(4)
        inplace[:2, :2] = [[c, -s], [s, c]]
        inplace[:2, 3] = -inplace[:2, :2] @ root
        inplace[2, 3] = float(a["ground_offset"])
        return cloud.transformed(inplace @ a["asset_to_standard"])


def _rigid_arrays(asset: LibraryAsset) -> dict[str, np.ndarray]:
    from plyfile import PlyData
    try:
        vertex = PlyData.read(str(asset.path))["vertex"].data
        names = vertex.dtype.names or ()
        def stack(fields):
            return np.stack([vertex[name] for name in fields], axis=-1).astype(np.float32)
        xyz = stack(["x", "y", "z"])
        q = unit_quaternions(stack([f"rot_{i}" for i in range(4)])).astype(np.float32)
        log_scales = stack([f"scale_{i}" for i in range(3)])
        scales = np.exp(log_scales)
        raw_opacity = np.asarray(vertex["opacity"], dtype=np.float64)
        # Stable sigmoid, including extreme legitimate logits.
        opacities = np.exp(-np.logaddexp(0, -raw_opacity)).astype(np.float32)
        sh = stack([f"f_dc_{i}" for i in range(3)])[:, None]
        rest = sorted((name for name in names if name.startswith("f_rest_")), key=lambda name: int(name[7:]))
        if rest:
            if len(rest) % 3 or rest != [f"f_rest_{i}" for i in range(len(rest))]:
                raise AssetError("Invalid PLY SH property layout")
            sh = np.concatenate((sh, stack(rest).reshape(len(xyz), 3, -1).transpose(0, 2, 1)), axis=1)
    except (OSError, ValueError, KeyError) as exc:
        raise AssetError(f"Cannot load Gaussian PLY {asset.path}: {exc}") from exc
    transform = np.diag([asset.meta["metric_scale"]] * 3 + [1.0])
    return dict(frame_xyz=xyz[None], frame_quaternions=q[None], frame_scales=scales[None],
                opacities=opacities, sh=sh, timestamps=np.zeros(1), root_xy=np.zeros((1, 2)),
                root_yaw=np.zeros(1), asset_to_standard=transform, ground_offset=np.array(0.0))


def _human_arrays(asset: LibraryAsset) -> dict[str, np.ndarray]:
    # Read only the exported coordinate representation, never both duplicate
    # frame collections or the native training runtime.
    with np.load(asset.path, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[source]) for key, source in {
            "frame_xyz": "frame_xyz", "frame_quaternions": "frame_rotation",
            "frame_scales": "frame_scaling", "opacities": "opacity", "sh": "shs",
            "timestamps": "frame_timestamps"}.items()}
        arrays["opacities"] = arrays["opacities"].reshape(-1)
        if bool(np.asarray(z["use_rgb"]).reshape(-1)[0]):
            rgb = arrays["sh"].reshape(-1, 3)
            arrays["sh"] = ((rgb - 0.5) / 0.28209479177387814)[:, None]
        else:
            degree = int(np.asarray(z["active_sh_degree"]).reshape(-1)[0])
            if arrays["sh"].shape[1] != (degree + 1) ** 2:
                raise AssetError("Human SH degree disagrees with coefficient count")
    with np.load(asset.meta["playback_path"], allow_pickle=False) as z:
        if not np.allclose(z["frame_timestamps"], arrays["timestamps"], atol=1e-8, rtol=0):
            raise AssetError("Human animation and placement clocks disagree")
        for key in ("root_xy", "root_yaw", "asset_to_standard", "ground_offset"):
            arrays[key] = np.asarray(z[key])
    return arrays


def load_asset(asset: LibraryAsset) -> RuntimeAsset:
    files = [*asset.source_files, asset.path]
    try:
        if asset.meta["format"] == "rigid_ply":
            arrays = _rigid_arrays(asset)
        elif asset.meta["format"] == "human_npz":
            arrays = _human_arrays(asset)
            files.append(Path(asset.meta["playback_path"]))
        elif asset.meta["format"] == _MINIMAL_FORMAT:
            with np.load(asset.path, allow_pickle=False) as z:
                arrays = {key: z[key] for key in z.files}
        else:
            raise AssetError(f"Unsupported asset format {asset.meta['format']!r}")
        return RuntimeAsset(asset, arrays, files)
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, AssetError):
            raise
        raise AssetError(f"Cannot load {asset.id}: {exc}") from exc


def export_minimal(asset: LibraryAsset, output: str | Path, *, cycle_profile=None) -> Path:
    """Create a standalone one-asset library; never alter the source delivery.

    The output contains only three runtime files. Quality limitations remain in
    metadata; the export changes packaging, not the claimed asset quality.
    """
    target = Path(output).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise AssetError(f"Minimal export destination is not empty: {target}")
    runtime = load_asset(asset)
    if cycle_profile is None and asset.meta.get("cycle_profile") is not None:
        from hasim.shared.assets.human_cycle import CycleProfile
        cycle_profile = CycleProfile.from_dict(asset.meta["cycle_profile"])
    if cycle_profile is not None:
        from hasim.shared.assets.human_cycle import CyclicRuntimeAsset
        # Verify source binding before writing a portable capability claim.
        CyclicRuntimeAsset(runtime, cycle_profile)
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "runtime.npz", **runtime.arrays)
    meta = dict(format=_MINIMAL_FORMAT, id=asset.id, category=asset.category, kind=asset.kind,
                payload="runtime.npz", duration_s=asset.duration_s if asset.kind == "human" else None,
                native_speed_mps=asset.native_speed_mps, motion_type=asset.motion_type,
                dimensions=asset.dimensions, quality_approved=bool(asset.meta.get("quality_approved", False)),
                quality_state=asset.meta.get("quality_state", "unknown"),
                producer_status=asset.meta.get("producer_status", asset.meta.get("status")),
                quality_flags=asset.meta.get("quality_flags", []),
                coordinate_convention="right_handed_x_forward_y_left_z_up",
                playback="once" if asset.kind == "human" else "static",
                metric_scale_verified=bool(asset.meta.get("metric_scale_verified", False)),
                playback_capabilities=asset.playback_capabilities,
                animation_sampling="baked_frames_linear_positions_and_spd_covariance")
    if cycle_profile is not None:
        meta["cycle_profile"] = cycle_profile.to_dict()
        meta["playback_capabilities"]["qualified_cycle_available"] = True
    (target / "asset.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (target / "library.json").write_text(json.dumps(dict(format=_MINIMAL_FORMAT, assets=[dict(metadata="asset.json")]), indent=2) + "\n", encoding="utf-8")
    return target
