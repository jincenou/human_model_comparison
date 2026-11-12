import unittest
import numpy as np
from human_edp_common import part_labels,pool_parts,compare_parts


class EdpContractTests(unittest.TestCase):
    def test_empty_common_parts_is_na(self):
        out=compare_parts(np.ones((6,3)),np.ones((6,3)),[1,0,0,0,0,0],[0,1,0,0,0,0])
        self.assertEqual(out['score']['status'],'na');self.assertEqual(out['coverage'],0)

    def test_identical_and_orthogonal_features(self):
        a=np.tile([1.,0.],(6,1));b=np.tile([0.,1.],(6,1))
        self.assertEqual(compare_parts(a,a,[1]*6,[1]*6)['score']['value'],0)
        self.assertEqual(compare_parts(a,b,[1]*6,[1]*6)['score']['value'],1)

    def test_missing_parts_keep_coverage(self):
        a=np.ones((6,2));out=compare_parts(a,a,[2]*6,[3,0,0,0,0,0])
        self.assertEqual(out['valid_parts'],['head']);self.assertEqual(out['coverage'],1/6)
        self.assertIsNone(out['parts']['torso']['distance'])

    def test_mask_and_confidence_bound_part_support(self):
        k=np.zeros((17,3));k[5]=[8,8,.9];k[6]=[24,8,.9]
        mask=np.full((32,32),255,np.uint8)
        labels=part_labels(mask,k);self.assertTrue((labels==1).all())
        mask[:,16:]=0;labels=part_labels(mask,k)
        self.assertTrue((labels[:,1]==-1).all())
        k[:,2]=0;self.assertTrue((part_labels(mask,k)==-1).all())

    def test_pool_only_own_part_patches(self):
        features=np.array([[[1.,0.],[100.,100.]]],np.float32)
        e,n=pool_parts(features,np.array([[0,-1]]))
        np.testing.assert_array_equal(e[0],[1,0]);self.assertEqual(n.tolist(),[1,0,0,0,0,0])

    def test_reference_occlusion_does_not_reassign_patches(self):
        k=np.zeros((17,3));k[0]=[8,8,.9];k[1]=[24,8,.9]
        k[5]=[8,24,.9];k[6]=[24,24,.9];mask=np.full((32,32),255,np.uint8)
        labels=part_labels(mask,k,excluded_parts=['head'])
        self.assertTrue((labels[0]==-1).all());self.assertTrue((labels[1]==1).all())


if __name__=='__main__':unittest.main()
