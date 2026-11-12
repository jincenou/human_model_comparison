"""Explain fixed source scores by spatial region; never modify scoring outputs."""
from pathlib import Path
import math
import time
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d, maximum_filter
from common_human_orbit import H,read,save,sha,checked
import human_source_metrics_v1 as metrics

OUT=H/'reports/source_metric_region_diagnostic_v1'


def ssim_map(a,b):
    a=a.astype(np.float64)/255;b=b.astype(np.float64)/255
    grid=np.arange(-5,6,dtype=np.float64);w=np.exp(-grid**2/(2*1.5**2));w/=w.sum()
    def smooth(x):return convolve1d(convolve1d(x,w,axis=0,mode='reflect'),w,axis=1,mode='reflect')
    ma,mb=smooth(a),smooth(b);va=smooth(a*a)-ma*ma;vb=smooth(b*b)-mb*mb;cov=smooth(a*b)-ma*mb
    return (((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))[5:-5,5:-5].mean(2)


def main():
    OUT.mkdir(exist_ok=True)
    path=H/'reports/source_camera_metric_cohort_v3/summary.json';summary=read(path)
    pairing=summary['pairings']['ours+ah+trellis_original+omnire+splatad']['PSNR'];ids=pairing['common_ids']
    rows={(r['id'],r['method']):r for r in summary['rows']};records=[]
    for sid in ids:
        ref=rows[(sid,'ours')]['reference']
        for k in ['reference_rgb','reference_mask','camera_contract']:checked(ref[k],ref[k+'_sha256'])
        reference=metrics.prepare_reference(np.array(Image.open(ref['reference_rgb']).convert('RGB')),np.array(Image.open(ref['reference_mask']).convert('L')))
        assert list(reference['roi'])==ref['roi_xyxy']
        target=reference['rgb'];fg=reference['mask']>127
        pure_background=~maximum_filter(fg,size=11,mode='constant')[5:-5,5:-5]
        row=dict(id=sid,reference=ref,foreground_fraction=float(fg.mean()),pure_background_window_fraction=float(pure_background.mean()),methods={})
        for method in ['ours','splatad']:
            scored=rows[(sid,method)];checked(scored['result'],scored['result_sha256']);result=read(scored['result'])
            checked(result['registration_record'],result['registration_sha256']);cert=read(result['registration_record'])
            assert cert['timestamp_us']==ref['timestamp_us'] and cert['camera_sha256']==ref['camera_contract_sha256']
            for k in ['source_rgb','source_alpha']:checked(cert[k],cert[k+'_sha256'])
            pred=metrics.prepare_prediction(np.array(Image.open(cert['source_rgb']).convert('RGB')),np.array(Image.open(cert['source_alpha']).convert('L')),reference['roi'],cert['prediction_encoding'])
            error=((pred['rgb'].astype(np.float64)-target.astype(np.float64))/255)**2
            mse=float(error.mean());ferr=float(error[fg].mean());berr=float(error[~fg].mean());fraction=float(fg.mean())
            assert abs(mse-(fraction*ferr+(1-fraction)*berr))<1e-14
            psnr=-10*math.log10(mse);np.testing.assert_allclose(psnr,result['metrics']['PSNR']['value'],atol=1e-10,rtol=0)
            structural=ssim_map(target,pred['rgb']);ssim=float(structural.mean())
            np.testing.assert_allclose(ssim,result['metrics']['SSIM']['value'],atol=1e-12,rtol=0)
            body=float(structural[~pure_background].mean());background=float(structural[pure_background].mean());bfrac=float(pure_background.mean())
            np.testing.assert_allclose(ssim,bfrac*background+(1-bfrac)*body,atol=1e-12,rtol=0)
            pmask=pred['mask']>127
            row['methods'][method]=dict(result=scored['result'],result_sha256=scored['result_sha256'],
                full_psnr=psnr,full_mse=mse,reference_foreground_mse=ferr,reference_background_mse=berr,
                reference_foreground_psnr=-10*math.log10(ferr),full_ssim=ssim,
                reference_body_and_boundary_ssim=body,reference_pure_background_ssim=background,
                lpips=result['metrics']['LPIPS']['value'],iou=result['metrics']['IoU']['value'],
                foreground_recall=float(np.count_nonzero(pmask&fg)/np.count_nonzero(fg)),
                foreground_precision=float(np.count_nonzero(pmask&fg)/max(1,np.count_nonzero(pmask))),
                reference_bg_error_share=float((1-fraction)*berr/mse),prediction_encoding=cert['prediction_encoding'])
        records.append(row)
    means={method:{key:float(np.mean([r['methods'][method][key]for r in records]))for key,value in records[0]['methods'][method].items()if isinstance(value,(int,float))}for method in ['ours','splatad']}
    report=dict(created_at=time.time(),summary=str(path),summary_sha256=sha(path),ids=ids,n=len(ids),rows=records,means=means,
        mean_foreground_fraction=float(np.mean([r['foreground_fraction']for r in records])),
        mean_pure_background_window_fraction=float(np.mean([r['pure_background_window_fraction']for r in records])),
        splatad_psnr_wins=sum(r['methods']['splatad']['full_psnr']>r['methods']['ours']['full_psnr']for r in records),
        same_source_scores_verified=True,core_metric_code_sha256=sha(metrics.__file__),script_sha256=sha(__file__),
        production_scores_modified=False,scope='Diagnostic spatial decomposition only; not replacement metrics or a new ranking',
        caveat='Region masks only index already-computed errors/maps. Predictions are never multiplied by ground-truth mask. Full scores keep false-positive background errors.')
    save(OUT/'diagnostic.json',report)
    print({k:v for k,v in report.items()if k not in ['rows','ids']})


if __name__=='__main__':main()
