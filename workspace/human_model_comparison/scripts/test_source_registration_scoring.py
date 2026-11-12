"""Check the declared rigid-registration route against the frozen image scorer."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import human_metric_batch_v1 as b
from run_human_metric_pipeline_v1 import check_active_registration, score_active_pair


class RegisteredSourceScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        active = b.read(b.H/'inputs/active_human_benchmark.json')
        q = b.read(active['reference_qualification'])
        cls.ref = next(r for r in q['objects'] if r['id'] == 'HN0004')
        p = b.H/'reports/trellis_source_registration/certificates/HN0004.json'
        cls.cert = b.read(p)
        cls.entry = dict(id='HN0004', method='trellis_original', reference=cls.ref,
                         registration_record=str(p), registration_sha256=b.m.sha(p), input_key='isolated-test-only')
        protocol = b.read(b.C/'core_metric_protocol_frozen_v1.json')
        b.verified(b.m.__file__, protocol['code_sha256'])
        cfg = protocol['lpips']; torch.set_num_threads(4)
        cls.metric = b.m.AlexLPIPS(cfg['backbone'], cfg['linear'], cfg['backbone_sha256'], cfg['linear_sha256'], 'cpu')

    def fixture(self, directory, cert):
        p = Path(directory)/'isolated_certificate.json'; b.save(p, cert)
        return dict(self.entry, registration_record=str(p), registration_sha256=b.m.sha(p))

    def test_identical_composition_roi_and_all_metrics(self):
        # Only in this temporary test fixture, switch the registration label to
        # exercise the old route on identical images. Never publish this fixture.
        plain = copy.deepcopy(self.cert); plain['posthoc_pose_or_scale_fit'] = False
        with tempfile.TemporaryDirectory() as td:
            old, old_ref, old_pred = b.score_pair(self.fixture(td, plain), self.metric)
        new, new_ref, new_pred = score_active_pair(self.entry, self.metric)
        self.assertEqual(old['metrics'], new['metrics'])
        for key in ['rgb', 'mask']:
            np.testing.assert_array_equal(old_ref[key], new_ref[key])
            np.testing.assert_array_equal(old_pred[key], new_pred[key])
        self.assertEqual(old_ref['roi'], new_ref['roi'])
        self.assertTrue(new['posthoc_pose_or_scale_fit'])
        self.assertFalse(new['real_heldout_eligible'])

    def test_score_guided_fit_and_camera_mismatch_rejected(self):
        for field, value in [('reference_metric_values_used', True), ('prediction_bbox_used', True),
                             ('timestamp_us', self.ref['timestamp_us']+1), ('camera_sha256', 'wrong')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                cert = dict(self.cert, **{field:value})
                with self.assertRaises(ValueError):
                    check_active_registration(self.ref, 'trellis_original', self.fixture(td, cert))

    def test_similarity_scale_cannot_be_mislabeled_rigid(self):
        with tempfile.TemporaryDirectory() as td:
            trace = b.read(self.cert['trace_record'])
            T = np.array(trace['T_source_camera_from_export']); T[:3,:3] *= 1.2
            trace['T_source_camera_from_export'] = T.tolist()
            p = Path(td)/'scaled_trace.json'; b.save(p, trace)
            cert = dict(self.cert, trace_record=str(p), trace_record_sha256=b.m.sha(p))
            with self.assertRaises(AssertionError):
                check_active_registration(self.ref, 'trellis_original', self.fixture(td, cert))


if __name__ == '__main__':
    unittest.main()
