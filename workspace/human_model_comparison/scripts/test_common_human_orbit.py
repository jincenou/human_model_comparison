"""Geometric invariants for shared cameras and native appearance coordinates."""
import unittest
import numpy as np
from common_human_orbit import camera_views,native_views


class OrbitGeometryTest(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[1200.,0,800.],[0,1250.,450.],[0,0,1.]])
        self.camera=dict(K_raw=self.K.tolist(),roi_xyxy=[450,300,850,700],T_world_from_camera_opencv=np.eye(4).tolist())
        self.center,self.up,self.frames=camera_views(self.camera)
        rng=np.random.default_rng(42);self.xyz=rng.normal(size=(100,3))*.15
        self.T=np.eye(4);self.T[:3,:3]=np.diag([-2.,2.,-2.]);self.T[:3,3]=[-.5,.2,8.]
    def project(self,xyz):
        p=xyz@self.K.T;return p[:,:2]/p[:,2:]
    def test_same_center_distance_and_angles(self):
        self.assertEqual([r['angle_degrees']for r in self.frames],[22.5*i for i in range(16)])
        np.testing.assert_allclose(self.frames[0]['camera_to_display'],np.eye(4),atol=1e-15)
        for f in self.frames:
            C=np.array(f['camera_to_display']);V=np.linalg.inv(C)
            np.testing.assert_allclose(C[:3,:3].T@C[:3,:3],np.eye(3),atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(C[:3,3]-self.center),np.linalg.norm(self.center))
            np.testing.assert_allclose(V[:3,:3]@self.center+V[:3,3],self.center,atol=1e-14)
    def test_source_projection_and_relative_pose_unchanged(self):
        views,g=native_views(self.xyz,self.T,self.frames);source=self.xyz@self.T[:3,:3].T+self.T[:3,3]
        native=self.xyz@views[0][:3,:3].T+views[0][:3,3]
        np.testing.assert_allclose(self.project(native),self.project(source),atol=1e-10)
        D=np.array(g['display_from_native'])
        self.assertAlmostEqual(np.median((self.xyz@D[:3,:3].T+D[:3,3])[:,2]),1.)
    def test_all_views_preserve_projection_and_native_view_direction(self):
        views,g=native_views(self.xyz,self.T,self.frames);D=np.array(g['display_from_native'])
        for f,V in zip(self.frames,views):
            C=np.array(f['camera_to_display']);combined=np.linalg.inv(C)@D
            np.testing.assert_allclose(self.project(self.xyz@V[:3,:3].T+V[:3,3]),self.project(self.xyz@combined[:3,:3].T+combined[:3,3]),atol=1e-10)
            # Native camera center is identical for the similarity and rigid
            # forms, so native SH/learned directional inputs do not get rotated twice.
            np.testing.assert_allclose(np.linalg.inv(V)[:3,3],np.linalg.inv(combined)[:3,3],atol=1e-12)
    def test_empty_or_behind_camera_geometry_is_technical_failure(self):
        with self.assertRaises(ValueError):native_views(np.zeros((0,3)),self.T,self.frames)
        T=np.eye(4);T[2,3]=-10
        with self.assertRaises(ValueError):native_views(self.xyz,T,self.frames)


if __name__=='__main__':unittest.main()
