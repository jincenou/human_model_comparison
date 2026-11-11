import dataclasses
import unittest
import numpy as np
from export_ours_frozen_time_v1 import H
from hasim.shared.assets.library_cloud import GaussianAssetFrame
from ours_coordinate_contract_v1 import evaluate,raster_difference,validate_raster

class CoordinateContractTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(371)
        q=rng.normal(size=(40,4));q/=np.linalg.norm(q,axis=1,keepdims=True)
        self.cloud=GaussianAssetFrame(rng.normal(size=(40,3)).astype('float32'),q.astype('float32'),
            np.tile(np.array([.02,.05,.12],dtype='float32'),(40,1)),np.ones(40,dtype='float32')*.7,
            rng.normal(size=(40,16,3)).astype('float32'),np.eye(3))
        self.camera=np.eye(4);self.camera[:3,3]=[3,4,5]
        self.rotation=np.eye(4);a=.73;self.rotation[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
        self.moved=self.cloud.transformed(self.rotation);self.new_camera=self.rotation@self.camera
    def passes(self,cloud):return evaluate(self.cloud,self.camera,cloud,self.new_camera)['passed']
    def test_correct_rigid_transform(self):self.assertTrue(self.passes(self.moved))
    def test_missing_covariance_rotation_rejected(self):
        self.assertFalse(self.passes(dataclasses.replace(self.moved,quaternions=self.cloud.quaternions)))
    def test_missing_appearance_rotation_rejected(self):
        self.assertFalse(self.passes(dataclasses.replace(self.moved,local_from_appearance=self.cloud.local_from_appearance)))
    def test_shifted_geometry_rejected(self):
        self.assertFalse(self.passes(dataclasses.replace(self.moved,means=self.moved.means+np.array([.001,0,0]))))
    def test_changed_sh_rejected(self):
        sh=self.moved.sh.copy();sh[0,8,1]+=.001
        self.assertFalse(self.passes(dataclasses.replace(self.moved,sh=sh)))
    def test_isolated_alpha_cutoff_vs_widespread_change(self):
        blank=np.zeros((512,512),dtype='float32');one=blank.copy();one[100,100]=1/255
        self.assertTrue(validate_raster(raster_difference(blank,one)))
        self.assertFalse(validate_raster(raster_difference(blank,blank+1/255)))
        one[100,100]=2/255
        self.assertFalse(validate_raster(raster_difference(blank,one)))
    def test_all_saved_real_snapshots_analytic(self):
        count=0
        for path in (H/'exports/ours_frozen_time_v1').glob('*/inplace.npz'):
            with np.load(path,allow_pickle=False)as z:
                cloud=GaussianAssetFrame(z['means'],z['quaternions_wxyz'],z['scales'],z['opacities'],z['sh'],z['local_from_appearance'])
            result=evaluate(cloud,self.camera,cloud.transformed(self.rotation),self.new_camera)
            self.assertTrue(result['passed'],(str(path),result));count+=1
        self.assertGreaterEqual(count,28)

if __name__=='__main__':unittest.main()
