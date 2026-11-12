"""Projection invariants and forbidden affine deformation for AH initialization."""
import unittest
import numpy as np
from prepare_ah_source_registration_pilot_v1 import initialization, project


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.view = np.eye(4)
        self.view[2,3] = 10
        self.kn = np.array([[2000,0,256],[0,2000,256],[0,0,1.]])
        self.kr = np.array([[1200,0,812],[0,1200,477],[0,0,1.]])
        self.crop = np.array([[2,0,-1200],[0,2,-600],[0,0,1.]])

    def test_origin_plane_projection_and_proper_scale(self):
        T = initialization(self.view,self.kn,self.kr,self.crop,1.7)
        points = np.array([[0,0,0],[.2,-.3,0],[-.4,.1,0]])
        native = project(points,self.view,self.kn)
        expected = (np.c_[native,np.ones(3)] @ np.linalg.inv(self.crop).T)[:,:2]
        np.testing.assert_allclose(project(points,T,self.kr),expected,atol=1e-10)
        np.testing.assert_allclose(np.linalg.svd(T[:3,:3],compute_uv=False),[1.7]*3)
        self.assertGreater(np.linalg.det(T[:3,:3]),0)

    def test_off_plane_is_not_claimed_projectively_equivalent(self):
        T = initialization(self.view,self.kn,self.kr,self.crop,1.7)
        points = np.array([[.2,-.3,.7]])
        native = project(points,self.view,self.kn)
        expected = (np.c_[native,np.ones(1)] @ np.linalg.inv(self.crop).T)[:,:2]
        self.assertGreater(float(np.abs(project(points,T,self.kr)-expected).max()),.1)

    def test_reject_anisotropic_focal_ratio(self):
        self.kr[1,1] = 1100
        with self.assertRaisesRegex(ValueError,'Anisotropic'):
            initialization(self.view,self.kn,self.kr,self.crop,1.7)

    def test_reject_reflection_and_nonpositive_scale(self):
        bad = self.view.copy();bad[0,0] = -1
        with self.assertRaises(ValueError):
            initialization(bad,self.kn,self.kr,self.crop,1.7)
        with self.assertRaises(ValueError):
            initialization(self.view,self.kn,self.kr,self.crop,0)


if __name__ == '__main__':
    unittest.main()
