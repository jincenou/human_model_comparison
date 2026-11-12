import unittest
import cv2
import numpy as np
import score_dynamic_source_clip as d


class DynamicMetricTests(unittest.TestCase):
    def test_flow_uses_vector_l2_and_tlp_compares_both_sequences(self):
        a = np.zeros((8, 8, 2), np.float32)
        b = np.empty_like(a); b[:] = (3, 4)
        result = d.temporal_pair(a, b, 0.2, 0.5)
        self.assertEqual(result['tOF']['value'], 5.0)
        self.assertAlmostEqual(result['tLP100']['value'], 30.0)
        self.assertEqual(d.temporal_pair(a, a, 0.5, 0.5)['tLP100']['value'], 0.0)

    def test_actual_flow_same_motion_zero_wrong_direction_nonzero(self):
        rng = np.random.default_rng(123)
        a = np.full((128, 128, 3), 255, np.uint8)
        a[24:104, 24:104] = rng.integers(0, 180, (80, 80, 3), np.uint8)
        b = cv2.warpAffine(a, np.array([[1, 0, 3], [0, 1, 0]], np.float32),
                          (128, 128), borderValue=(255,255,255))
        correct = d.optical_flow(a, b)
        self.assertEqual(d.flow_difference(correct, d.optical_flow(a, b))['value'], 0)
        self.assertGreater(d.flow_difference(correct, d.optical_flow(b, a))['value'], 0.1)

    def test_real_timestamps_are_not_assumed_uniform_and_cannot_repeat(self):
        frames = [dict(index=i, timestamp_us=t) for i,t in enumerate([1000000,1100000,1150000])]
        self.assertEqual(d.check_times(frames), [0.1,0.05])
        with self.assertRaises(ValueError):
            d.check_times([frames[0], dict(index=1,timestamp_us=1000000)])
        with self.assertRaises(ValueError):
            d.check_times([frames[0], dict(index=2,timestamp_us=1100000)])

    def test_predictions_are_not_erased_by_reference_mask(self):
        ref = np.full((512,512,3),255,np.uint8)
        ref[180:330,220:290] = 100
        mask = np.zeros((512,512),np.uint8); mask[180:330,220:290] = 255
        wrong = ref.copy(); wrong[200:260,300:340] = 0
        alpha = mask.copy();alpha[200:260,300:340] = 255
        pred = d.core.prepare_prediction(wrong,alpha,(0,0,512,512),'white_composited_uint8')
        self.assertTrue(np.all(pred['rgb'][210:250,310:330] == 0))
        self.assertLess(d.core.iou(mask,pred['mask'])['value'],1)
        self.assertEqual(d.core.psnr(ref,pred['rgb'])['status'],'ok')

    def test_empty_missing_scores_remain_na(self):
        self.assertEqual(d.mean_metric([])['status'],'na')
        self.assertEqual(d.mean_metric([d.core.score(1),d.core.na('missing')])['status'],'na')
        self.assertEqual(d.mean_metric([d.core.score(float('inf'))])['status'],'positive_infinity')


if __name__ == '__main__':
    unittest.main()
