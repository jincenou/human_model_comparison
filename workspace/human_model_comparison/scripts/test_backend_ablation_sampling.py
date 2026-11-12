"""Check physical covariance interpolation and source-clock boundaries."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from backend_ablation_sampling import source_bracket, blend_exported_frames


def covariance(q, s):
    r = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    return (r*s[:, None, :]**2) @ r.transpose(0, 2, 1)


class SamplingTests(unittest.TestCase):
    def test_fractional_exposure_is_not_rounded(self):
        lo, hi, f = source_bracket(dict(source_time_seconds=1.350009,
                                       bracket=[33, 34], fraction=.750225), 61)
        self.assertEqual((lo, hi), (33, 34))
        self.assertAlmostEqual((lo+f*(hi-lo))/25, 1.350009, places=14)

    def test_endpoint_and_out_of_range(self):
        self.assertEqual(source_bracket(dict(source_time_seconds=2.4,
                                            bracket=[59, 60], fraction=1.), 61), (59, 60, 1.))
        for t in (-.001, 2.401, float('nan')):
            with self.assertRaises(ValueError):
                source_bracket(dict(source_time_seconds=t, bracket=[59, 60], fraction=1.), 61)

    def test_stale_bracket_is_rejected(self):
        with self.assertRaises(ValueError):
            source_bracket(dict(source_time_seconds=1.350009, bracket=[34, 35], fraction=.75), 61)

    def test_eigenframe_sign_change_does_not_invent_motion(self):
        left = (np.array([[0., 1., 2.]]), np.array([[1., 0., 0., 0.]]), np.array([[.1, .3, .9]]))
        right = (np.array([[4., 3., 2.]]), np.array([[0., 0., 0., 1.]]), left[2])
        x, q, s = blend_exported_frames(left, right, .25)
        np.testing.assert_allclose(x, [[1., 1.5, 2.]])
        np.testing.assert_allclose(covariance(q, s), covariance(left[1], left[2]), atol=1e-7)

    def test_anisotropic_mixture_matches_physical_covariance(self):
        left = (np.zeros((1, 3)), np.array([[1., 0., 0., 0.]]), np.array([[.1, .3, .9]]))
        q = Rotation.from_euler('xyz', [[30, 40, 80]], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        right = (np.ones((1, 3)), q, np.array([[.7, .2, .4]]))
        for f in (0., .25, .75, 1.):
            _, mixed_q, mixed_s = blend_exported_frames(left, right, f)
            expected = (1-f)*covariance(left[1], left[2])+f*covariance(right[1], right[2])
            np.testing.assert_allclose(covariance(mixed_q, mixed_s), expected, atol=1e-7)
        for f, endpoint in ((0., left), (1., right)):
            for actual, expected in zip(blend_exported_frames(left, right, f), endpoint):
                np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
