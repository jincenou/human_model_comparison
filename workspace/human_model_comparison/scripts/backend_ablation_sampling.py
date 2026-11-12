"""Match production baked-frame sampling at an exact source exposure.

The deformation network is evaluated at the two original 25 Hz frame times.
Interpolate converted physical covariance using the existing asset runtime;
evaluating the network directly at a fractional time would be a different rule.
"""
import numpy as np

# This module registers the same runtime package used for production sampling.
from export_ours_frozen_time_v1 import SRC
from hasim.shared.assets.library_cloud import interpolate_covariances


def source_bracket(registration, frame_count):
    times = np.arange(frame_count, dtype=np.float64) / 25
    query = float(registration['source_time_seconds'])
    if not np.isfinite(query) or query < 0 or query > times[-1]:
        raise ValueError('Source exposure outside the non-looping native clip')
    hi = min(int(np.searchsorted(times, query, side='right')), frame_count - 1)
    lo = max(0, hi - 1)
    fraction = 0. if hi == lo else float((query-times[lo])/(times[hi]-times[lo]))
    if [lo, hi] != registration['bracket']:
        raise ValueError('Registration and native source-clock bracket differ')
    if abs(fraction-float(registration['fraction'])) > 1e-12:
        raise ValueError('Registration and native interpolation fraction differ')
    return lo, hi, fraction


def blend_exported_frames(left, right, fraction):
    """Endpoints are (positions, wxyz quaternions, positive scales)."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid interpolation fraction')
    for a, b in zip(left, right):
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid or incompatible native frames')
    q, s = interpolate_covariances(left[1], left[2], right[1], right[2], fraction)
    return (1-fraction)*left[0]+fraction*right[0], q, s
