"""Incremental bookkeeping tests use synthetic pixels and mock LPIPS only.

These tests do not certify the geometry of a real method's registration.
"""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import human_metric_batch_v1 as b

class BatchTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.h=Path(self.temp.name);self.c=self.h/'cfg';self.c.mkdir();self.run=self.h/'run';self.run.mkdir()
  self.patch1=patch.object(b,'H',self.h);self.patch2=patch.object(b,'C',self.c);self.patch1.start();self.patch2.start();self.addCleanup(self.patch1.stop);self.addCleanup(self.patch2.stop)
  rgb=np.full((64,64,3),255,np.uint8);rgb[20:44,24:40]=80;mask=np.zeros((64,64),np.uint8);mask[20:44,24:40]=255
  Image.fromarray(rgb).save(self.h/'rgb.png');Image.fromarray(mask).save(self.h/'mask.png');self.refs=[]
  for i in range(100):
   sid=f'HN{i:04d}';cam=self.c/(sid+'.json');b.save(cam,dict(render_wh=[64,64],roi_xyxy=list(b.m.reference_roi(mask))))
   self.refs.append(dict(id=sid,identity_key='nuscenes:'+sid,timestamp_us=i,cohort='synthetic',reference_ready=True,reference_eligible=True,na_reasons=[],camera_contract=str(cam),camera_contract_sha256=b.m.sha(cam),reference_rgb=str(self.h/'rgb.png'),reference_rgb_sha256=b.m.sha(self.h/'rgb.png'),reference_mask=str(self.h/'mask.png'),reference_mask_sha256=b.m.sha(self.h/'mask.png'),roi_xyxy=list(b.m.reference_roi(mask)),reference_pixel_height=24))
  selection=self.h/'selection.json';b.save(selection,dict(objects=[dict(id=r['id']) for r in self.refs]));self.pp=self.c/'core_metric_protocol_frozen_v1.json';b.save(self.pp,dict(code_sha256=b.m.sha(b.m.__file__),lpips=dict(backbone='mock',linear='mock',backbone_sha256='mock',linear_sha256='mock')))
  self.qp=self.c/'reference_qualification_frozen_v1.json';b.save(self.qp,dict(objects=self.refs,denominator=100,selection=str(selection),selection_sha256=b.m.sha(selection),core_protocol_sha256=b.m.sha(self.pp)))
  render=self.h/'render.json';script=self.h/'renderer.py';b.save(render,{'status':'passed'});script.write_text('# synthetic test fixture\n')
  r=self.refs[1];self.reg=self.h/'registration.json';b.save(self.reg,dict(status='passed',registration_certified=True,identity=r['id'],identity_key=r['identity_key'],timestamp_us=r['timestamp_us'],camera_sha256=r['camera_contract_sha256'],method='omnire',posthoc_pose_or_scale_fit=False,reference_metric_values_used=False,prediction_encoding='white_composited_uint8',source_rgb=r['reference_rgb'],source_rgb_sha256=r['reference_rgb_sha256'],source_alpha=r['reference_mask'],source_alpha_sha256=r['reference_mask_sha256'],render_validation=str(render),render_validation_sha256=b.m.sha(render),render_script=str(script),render_script_sha256=b.m.sha(script)))
  self.index=self.h/'index.json';self.refresh_index()
 def refresh_index(self):
  b.save(self.index,dict(qualification=str(self.qp),qualification_sha256=b.m.sha(self.qp),rows=[dict(id='HN0001',method='omnire',status='certified_for_core_metrics',registration_record=str(self.reg),registration_sha256=b.m.sha(self.reg))]))
 def prepare(self):return b.prepare(self.run,['omnire'],[self.index])
 def test_prepare_idempotent_and_no_missing_scores(self):
  p=self.prepare();a=b.read(self.run/'latest_plan.json')['key'];self.prepare();self.assertEqual(a,b.read(self.run/'latest_plan.json')['key']);self.assertEqual(p['summary']['omnire'],{'awaiting_asset':99,'ready':1})
 def test_reference_only_policy_rejects_pending(self):
  q=b.read(self.qp);q['objects'][1].update(reference_eligible=False,na_reasons=['external_occlusion']);b.save(self.qp,q);self.refresh_index();p=self.prepare();self.assertEqual(p['rows'][1]['status'],'reference_na')
 def test_identity_time_camera_mismatch_rejected(self):
  for field,value in [('identity','other'),('timestamp_us',999),('camera_sha256','bad')]:
   old=b.read(self.reg);bad=dict(old);bad[field]=value;b.save(self.reg,bad);self.refresh_index();p=self.prepare();self.assertEqual(p['rows'][1]['status'],'validation_failed');b.save(self.reg,old)
 def test_changed_pixels_rejected_before_prepare(self):
  Image.fromarray(np.zeros((64,64,3),np.uint8)).save(self.h/'rgb.png')
  with self.assertRaisesRegex(ValueError,'Changed input'):self.prepare()
 def test_cohort_not_silently_reduced(self):
  q=b.read(self.qp);q['objects'].pop();b.save(self.qp,q);self.refresh_index()
  with self.assertRaises(AssertionError):self.prepare()
 def test_incremental_run_reuses_success(self):
  self.prepare()
  class LPIPS:
   provenance={'synthetic_mock':True}
   def __call__(self,a,c):return b.m.score(0)
  with patch.object(b.m,'AlexLPIPS',return_value=LPIPS()) as factory:
   b.run_scores(self.run);self.assertEqual(factory.call_count,1)
   b.run_scores(self.run);self.assertEqual(factory.call_count,1)
  s=b.read(self.run/'state.json');self.assertEqual(s['reused'],1);self.assertEqual(s['scored_new'],0);self.assertEqual(b.read(self.run/'summary.json')['coverage']['omnire'],1)
 def test_registration_revision_gets_new_result_key(self):
  a=self.prepare()['rows'][1]['input_key'];r=b.read(self.reg);r['revision']='new certified record';b.save(self.reg,r);self.refresh_index();z=self.prepare()['rows'][1]['input_key'];self.assertNotEqual(a,z)
if __name__=='__main__':unittest.main()
