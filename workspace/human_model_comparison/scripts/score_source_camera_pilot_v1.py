"""Diagnostic numeric pilot; never promote unregistered preview images to scores."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
import human_source_metrics_v1 as m

H=Path(__file__).resolve().parents[1]
OUT=H/'reports/source_camera_metric_pilot_v1'
IDS=['HN0035','HN0039','HN0040']
METHODS=['ours','ah','trellis_original','omnire','splatad']

def main():
    torch.set_num_threads(4)
    cfgpath=H/'configs/human_source_common_v1/metric_protocol_candidate_v1.json'
    cfg=json.loads(cfgpath.read_text())['lpips']
    lpips=m.AlexLPIPS(cfg['backbone'],cfg['linear'],cfg['backbone_sha256'],cfg['linear_sha256'])
    OUT.mkdir(exist_ok=False)
    rows=[]
    for sid in IDS:
        cp=H/'configs/human_source_common_v1/reviewed_source_v2'/sid/'camera.json'
        c=json.loads(cp.read_text());assert not c['reference_pending_reasons']
        assert m.sha(c['source_rgb'])==c['source_rgb_sha256'] and m.sha(c['source_mask'])==c['source_mask_sha256']
        raw=np.asarray(Image.open(c['source_rgb']).convert('RGB'));mask=np.asarray(Image.open(c['source_mask']).convert('L'))
        ref=m.prepare_reference(raw,mask);assert list(ref['roi'])==c['roi_xyxy']
        for method in METHODS:
            row=dict(id=sid,method=method,metric_eligible=False,formal_scored=False,
                     scope='source-camera numerical pilot; registration certification and full metric protocol pending',
                     camera_contract=str(cp),camera_contract_sha256=m.sha(cp),metrics={})
            if method not in ('omnire','splatad'):
                row.update(status='not_scored',reason='source_camera_registration_not_yet_validated; native preview is not a score input')
                rows.append(row);continue
            p=H/'exports/common_camera_human_original75_time_v1'/method/sid/'validation.json'
            v=json.loads(p.read_text());a=v['common_camera_audit']
            assert v['status']=='passed' and v['state_time_query_valid'] and v['stored_canonical_parameters_equal']
            assert v['frozen_raw_timestamp_us']==c['timestamp_us'] and v['selection_sha256']==c['selection_sha256']
            assert Path(v['source_image']).resolve()==Path(c['source_rgb']).resolve()
            np.testing.assert_allclose(a['K_raw'],c['K_raw'],atol=1e-9,rtol=0)
            np.testing.assert_allclose(a['T_world_from_camera_opencv'],c['T_world_from_camera_opencv'],atol=1e-9,rtol=0)
            assert a['render_wh']==c['render_wh'] and a['projection_max_error_px']<.01
            pred_rgb=np.asarray(Image.open(v['actor_preview']).convert('RGB'))
            pred_alpha=np.asarray(Image.open(v['actor_alpha']).convert('L'))
            assert pred_rgb.shape==raw.shape and pred_alpha.shape==mask.shape
            # Both renderer sources explicitly composite native foreground using own alpha onto white.
            pred=m.prepare_prediction(pred_rgb,pred_alpha,ref['roi'],'white_composited_uint8')
            scores={'PSNR':m.psnr(ref['rgb'],pred['rgb']),'SSIM':m.ssim(ref['rgb'],pred['rgb']),
                    'LPIPS':lpips(ref['rgb'],pred['rgb']),'IoU':m.iou(ref['mask'],pred['mask']),
                    'IoU_raw':m.iou(mask,pred_alpha),'ED-P':m.na('feature_extractor_and_visibility_protocol_pending')}
            folder=OUT/sid/method;folder.mkdir(parents=True)
            for name,array in [('reference',ref['rgb']),('prediction',pred['rgb']),('reference_mask',ref['mask']),('prediction_alpha',pred['mask'])]:
                Image.fromarray(array).save(folder/(name+'.png'))
            row.update(status='diagnostic_scored',metrics=scores,roi_xyxy=list(ref['roi']),
                       render_validation=str(p),render_validation_sha256=m.sha(p),
                       prediction_sha256=m.sha(v['actor_preview']),alpha_sha256=m.sha(v['actor_alpha']),
                       projection_max_error_px=a['projection_max_error_px'],
                       reference_pixel_height=c['reference_pixel_height'],prediction_encoding='white_composited_uint8')
            rows.append(row)
            print(sid,method,json.dumps(scores),flush=True)
    result=dict(status='diagnostic_pilot_complete_not_formal_comparison',
                pilot_selection='Fixed original75 pilot IDs HN0035/HN0039/HN0040, inherited from extraction pilots; no score selection',
                methods_requested=METHODS,pilot_ids=IDS,selected_denominator=100,
                metric_code_sha256=m.sha(Path(m.__file__)),protocol_sha256=m.sha(cfgpath),
                lpips=lpips.provenance,rows=rows,rankings_produced=False,
                remaining=['source registration for Ours/AH/TRELLIS','scene registration certification','reference eligibility freeze','ED-P feature/visibility implementation','shared orbit validation'])
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
