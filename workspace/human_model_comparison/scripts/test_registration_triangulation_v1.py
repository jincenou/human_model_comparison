import unittest
import numpy as np
from analyze_registration_keypoints_v1 import triangulate
from prepare_trellis_keypoint_probe_v1 import native_camera


def pixels(cameras,point):
    out=[]
    for c in cameras:
        p=np.asarray(c['projection_matrix']) @ np.r_[point,1.]
        out.append(p[:2]/p[2])
    return np.array(out)


class TriangulationTests(unittest.TestCase):
    def test_recovers_nonplanar_points_in_native_coordinates(self):
        cams=[native_camera(y) for y in [0,.3,1.2,2.8]]
        for point in [[.12,-.2,.1],[-.13,.08,-.07],[0,0,.35]]:
            r=triangulate(cams,pixels(cams,point),[.7,.8,.6,.9])
            np.testing.assert_allclose(r['xyz'],point,atol=1e-10)
            self.assertLess(r['max_reprojection_error_pixels'],1e-9)

    def test_identical_cameras_rejected(self):
        cam=native_camera(0);cams=[cam,cam]
        with self.assertRaises(ValueError):triangulate(cams,pixels(cams,[.1,.1,.1]))

    def test_bad_correspondence_is_visible_in_residuals(self):
        cams=[native_camera(y) for y in [0,.7,1.4]]
        observed=pixels(cams,[.1,-.1,.1]);observed[0]+=[30,20]
        r=triangulate(cams,observed)
        self.assertGreater(r['max_reprojection_error_pixels'],5)

    def test_negative_weights_rejected(self):
        cams=[native_camera(0),native_camera(.7)]
        with self.assertRaises(ValueError):triangulate(cams,pixels(cams,[.1,0,.1]),[1,-1])


if __name__=='__main__':unittest.main()
