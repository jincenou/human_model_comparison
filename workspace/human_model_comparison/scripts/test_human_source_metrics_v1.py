import json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from skimage.metrics import structural_similarity
import human_source_metrics_v1 as m
H=Path(__file__).resolve().parents[1]

class SourceMetricTest(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(32)
        self.rgb=self.rng.integers(0,256,(64,64,3),dtype=np.uint8)
        self.mask=np.zeros((64,64),dtype=np.uint8);self.mask[18:46,18:46]=255
    def test_identity_and_infinity_serialization(self):
        self.assertEqual(m.psnr(self.rgb,self.rgb)['status'],'positive_infinity')
        self.assertEqual(m.ssim(self.rgb,self.rgb)['value'],1)
        self.assertEqual(m.iou(self.mask,self.mask)['value'],1)
        json.dumps(m.psnr(self.rgb,self.rgb),allow_nan=False)
    def test_known_psnr(self):
        a=np.zeros((16,16,3),dtype=np.uint8);b=np.full_like(a,255)
        self.assertEqual(m.psnr(a,b)['value'],0.)
        b.fill(1);self.assertAlmostEqual(m.psnr(a,b)['value'],20*np.log10(255),places=12)
    def test_ssim_independent_library(self):
        other=np.roll(self.rgb,3,axis=1)
        expected=structural_similarity(self.rgb.astype(np.float64)/255,other.astype(np.float64)/255,
            channel_axis=2,data_range=1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,win_size=11,K1=.01,K2=.03)
        self.assertAlmostEqual(m.ssim(self.rgb,other)['value'],expected,places=12)
    def test_empty_prediction_is_measured(self):
        ref=m.prepare_reference(self.rgb,self.mask)
        pred=m.prepare_prediction(np.zeros_like(self.rgb,dtype=np.float64),np.zeros_like(self.mask,dtype=np.float64),ref['roi'],'straight_float')
        self.assertEqual(m.iou(ref['mask'],pred['mask'])['value'],0)
        self.assertEqual(m.psnr(ref['rgb'],pred['rgb'])['status'],'ok')
        self.assertTrue(np.all(pred['rgb']==255))
    def test_empty_reference_is_na(self):
        mask=np.zeros_like(self.mask)
        self.assertEqual(m.prepare_reference(self.rgb,mask)['status'],'na')
        self.assertEqual(m.iou(mask,mask)['status'],'na')
    def test_wrong_structure_outside_gt_is_not_erased(self):
        ref=m.prepare_reference(np.zeros_like(self.rgb),self.mask)
        alpha=self.mask.astype(np.float64)/255;alpha[17:18,18:46]=1
        pred=m.prepare_prediction(np.zeros_like(self.rgb,dtype=np.float64),alpha,ref['roi'],'straight_float')
        outside=ref['mask']==0
        self.assertTrue((pred['rgb'][outside]<200).any())
        self.assertLess(m.iou(ref['mask'],pred['mask'])['value'],1)
    def test_deliberate_displacement(self):
        pred=np.roll(self.mask,9,axis=1)
        self.assertLess(m.iou(self.mask,pred)['value'],1)
    def test_border_padding_and_reference_roi(self):
        rgb=np.zeros((16,16,3),dtype=np.uint8);mask=np.zeros((16,16),dtype=np.uint8);mask[:10,:10]=255
        ref=m.prepare_reference(rgb,mask)
        self.assertEqual(ref['roi'],(-1,-1,10,10))
        self.assertTrue(np.all(ref['rgb'][0,0]==255));self.assertEqual(ref['mask'][0,0],0)
    def test_alpha_boundary_and_encoding_agreement(self):
        self.assertEqual(m.quantize(np.array([.5-1e-8,.5,.5+1e-8])).tolist(),[127,128,128])
        rgb=np.full((64,64,3),.4);alpha=self.mask.astype(np.float64)/255*.6;roi=m.reference_roi(self.mask)
        a=m.prepare_prediction(rgb,alpha,roi,'straight_float');b=m.prepare_prediction(rgb*alpha[...,None],alpha,roi,'premultiplied_float')
        np.testing.assert_array_equal(a['rgb'],b['rgb']);np.testing.assert_array_equal(a['mask'],b['mask'])
    def test_empty_edp_parts(self):
        r=m.aggregate_visible_parts({p:None for p in m.PARTS})
        self.assertEqual(r['score']['status'],'na');self.assertEqual(r['coverage'],0)
    def test_pairing_uses_same_people_and_full_denominator(self):
        def row(v,eligible=True):return dict(metric_eligible=eligible,metrics={'PSNR':m.score(v)})
        a={'a':row(40),'b':row(20),'c':row(50,False)};b={'b':row(30),'c':row(60)}
        r=m.paired_summary({'A':a,'B':b},'PSNR',['a','b','c','d'])
        self.assertEqual(r['common_ids'],['b']);self.assertEqual(r['paired']['A']['mean']['value'],20)
        self.assertEqual(r['paired']['B']['mean']['value'],30);self.assertEqual(r['paired']['A']['denominator'],4)
        self.assertEqual(r['individual_coverage']['A']['n'],2)

class LPIPSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(4)
        cls.cfg=json.loads((H/'configs/human_source_common_v1/metric_protocol_candidate_v1.json').read_text())['lpips']
        c=cls.cfg;cls.metric=m.AlexLPIPS(c['backbone'],c['linear'],c['backbone_sha256'],c['linear_sha256'])
    def test_identity_and_official_loader_agreement(self):
        import torch,lpips
        rng=np.random.default_rng(19);a=rng.integers(0,256,(64,64,3),dtype=np.uint8);b=np.roll(a,4,axis=0)
        self.assertLess(abs(self.metric(a,a)['value']),1e-8)
        old_dir=torch.hub.get_dir();torch.hub.set_dir(str(Path(self.cfg['backbone']).parent.parent))
        try:
            with patch.object(torch.hub,'download_url_to_file',side_effect=RuntimeError('Network forbidden in validation')):
                official=lpips.LPIPS(net='alex',version='0.1',model_path=self.cfg['linear'],verbose=False).eval()
            to_tensor=lambda x:torch.from_numpy(x.astype(np.float32)/127.5-1).permute(2,0,1)[None]
            with torch.no_grad():expected=float(official(to_tensor(a),to_tensor(b)).item())
            self.assertGreater(expected,0);self.assertAlmostEqual(self.metric(a,b)['value'],expected,places=8)
        finally:torch.hub.set_dir(old_dir)

if __name__=='__main__':unittest.main()
