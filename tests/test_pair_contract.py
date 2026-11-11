import importlib.util
from pathlib import Path
import tempfile
import unittest
from human_asset_benchmark.evaluate import evaluate

spec=importlib.util.spec_from_file_location('fixture_builder',Path(__file__).parents[1]/'tools/make_synthetic_example.py')
fixture_builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_builder)

class PairContractTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.pair=fixture_builder.make_example(Path(self.temp.name)/'example')
    def test_exact_pair(self):
        result=evaluate(self.pair)
        self.assertEqual(result['metrics']['PSNR']['status'],'positive_infinity')
        self.assertEqual(result['metrics']['SSIM']['value'],1)
        self.assertEqual(result['metrics']['IoU']['value'],1)
        self.assertEqual(result['metrics']['ED-P']['status'],'na')
    def test_reject_different_person(self):
        self.pair['prediction']['identity']='another-person'
        with self.assertRaisesRegex(ValueError,'mismatch'):evaluate(self.pair)
    def test_reject_different_time(self):
        self.pair['prediction']['timestamp_us']=2
        with self.assertRaisesRegex(ValueError,'mismatch'):evaluate(self.pair)
    def test_reject_different_camera(self):
        self.pair['prediction']['camera_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'mismatch'):evaluate(self.pair)
    def test_reject_unregistered_preview(self):
        self.pair['prediction']['registration_validated']=False
        with self.assertRaisesRegex(ValueError,'not been validated'):evaluate(self.pair)
    def test_reject_changed_image(self):
        p=Path(self.pair['prediction']['rgb']);p.write_bytes(p.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):evaluate(self.pair)
    def test_reject_reference_without_eligibility(self):
        self.pair['reference_eligible']=False
        with self.assertRaisesRegex(ValueError,'eligibility'):evaluate(self.pair)
