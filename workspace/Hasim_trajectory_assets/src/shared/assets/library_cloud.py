"""Backend-independent, activated Gaussian frames for curated asset playback.

Geometry is in right-handed X-forward/Y-left/Z-up, root-local model metres.
SH coefficients remain in their original appearance frame. For a row direction
``d`` in geometry coordinates evaluate SH at ``d @ local_from_appearance``.
This preserves every SH band without baking a new SH approximation each frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from hasim.shared.contracts.errors import AssetError


def unit_quaternions(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.any(norms < 1e-10):
        raise AssetError("Gaussian quaternions must be finite, nonzero wxyz")
    return q / norms


def slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    """Vectorized shortest-arc quaternion interpolation, including q/-q."""
    a, b = unit_quaternions(left), unit_quaternions(right)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0, 1)
    theta = np.arccos(dot)
    denominator = np.sin(theta)
    safe = np.maximum(denominator, 1e-12)
    spherical = (np.sin((1 - fraction) * theta) * a + np.sin(fraction * theta) * b) / safe
    linear = (1 - fraction) * a + fraction * b
    return unit_quaternions(np.where(dot > 0.9995, linear, spherical)).astype(np.float32)


def interpolate_covariances(left_q: np.ndarray, left_scale: np.ndarray,
                           right_q: np.ndarray, right_scale: np.ndarray,
                           fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate physical Gaussian covariance, independent of eigenframe signs.

    An anisotropic Gaussian has several equivalent quaternion/scale frames:
    flipping two eigenvector signs rotates its frame by pi without changing
    the covariance. SLERP between such frames invents a deformation. Interpolate
    the SPD matrices instead, then factor the result for Gaussian rasterizers.
    Endpoints retain their supplied representation exactly.
    """
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise AssetError("Gaussian interpolation fraction must be in [0, 1]")
    if fraction == 0.:
        return np.asarray(left_q), np.asarray(left_scale)
    if fraction == 1.:
        return np.asarray(right_q), np.asarray(right_scale)

    def covariance(quaternions, scales):
        q = unit_quaternions(quaternions)
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (*q.shape[:-1], 3) or np.any(scales <= 0) or not np.isfinite(scales).all():
            raise AssetError("Gaussian covariance requires matching positive finite scales")
        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        return (rotations * scales[:, None, :] ** 2) @ rotations.transpose(0, 2, 1)

    first, second = covariance(left_q, left_scale), covariance(right_q, right_scale)
    if first.shape != second.shape:
        raise AssetError("Gaussian covariance endpoint shapes must agree")
    mixed = (1 - fraction) * first + fraction * second
    eigenvalues, axes = np.linalg.eigh(mixed)
    # Rounding at extreme anisotropy may produce a tiny nonpositive eigenvalue;
    # floor far below any meaningful metric scale, without changing the major axes.
    scales = np.sqrt(np.maximum(eigenvalues, np.finfo(np.float32).tiny)).astype(np.float32)
    axes[:, :, -1] *= np.where(np.linalg.det(axes) < 0, -1., 1.)[:, None]
    quaternions = Rotation.from_matrix(axes).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
    return quaternions, scales


def multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product; composition applies right then left."""
    a, b = np.asarray(left), np.asarray(right)
    w = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    xyz = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, xyz), axis=-1)


def similarity(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise AssetError("Expected a finite 4x4 asset similarity")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise AssetError("Asset similarity must have an affine last row")
    scale = float(np.linalg.norm(transform[:3, :3], axis=0).mean())
    if scale <= 0:
        raise AssetError("Asset scale must be positive")
    rotation = transform[:3, :3] / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5):
        raise AssetError("Asset transform must be an orientation-preserving uniform similarity")
    # Saved float32 calibration may have tiny orthogonality error. Project onto SO(3).
    rotation = Rotation.from_matrix(rotation).as_matrix()
    return scale, rotation


@dataclass(frozen=True)
class GaussianAssetFrame:
    means: np.ndarray
    quaternions: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray
    local_from_appearance: np.ndarray

    @property
    def sh_degree(self) -> int:
        return int(round(np.sqrt(self.sh.shape[1]))) - 1

    def transformed(self, world_from_local: np.ndarray) -> "GaussianAssetFrame":
        """Transform geometry and carry the appearance frame into the world."""
        transform = np.asarray(world_from_local, dtype=np.float64)
        scale, rotation = similarity(transform)
        xyzw = Rotation.from_matrix(rotation).as_quat()
        q = xyzw[[3, 0, 1, 2]]
        # Match the proper rotation used for covariance and SH, including its
        # orthogonality correction, rather than the rounded input linear block.
        means = self.means @ (scale * rotation).T + transform[:3, 3]
        return GaussianAssetFrame(
            means.astype(np.float32),
            unit_quaternions(multiply_quaternions(q, self.quaternions)).astype(np.float32),
            (self.scales * scale).astype(np.float32),
            self.opacities,
            self.sh,
            rotation @ self.local_from_appearance,
        )

    def appearance_directions(self, local_directions: np.ndarray) -> np.ndarray:
        """Map normalized row directions to the unchanged SH coordinate frame."""
        return np.asarray(local_directions) @ self.local_from_appearance
