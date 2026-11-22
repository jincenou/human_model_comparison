"""Read-only arithmetic audit and scale-support diagnostic for new geometry."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

H=Path(__file__).resolve().parents[1]
ROOT=H/'reports/native_human_export_reaudit_v1'
GEOM=ROOT/'geometry'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dist_means(P,Q):
    left=[];right=np.full(len(Q),np.inf)
    for start in range(0,len(P),256):
        D=cdist(P[start:start+256],Q);left.extend(D.min(1));right=np.minimum(right,D.min(0))
    return np.asarray(left),right


def quantiles(values):
    if not values:return dict(n=0)
    return dict(n=len(values),**dict(zip(['min','p05','p25','median','p75','p95','max'],map(float,np.quantile(values,[0,.05,.25,.5,.75,.95,1])))))


def main():
    refs={r['id']:r for r in read(H/'evaluation_static_cd_fscore_time_v1/reference_manifest.json')['rows']}
    result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='diagnostic_only_not_final_model_geometry_comparison',methods={},
        fixed_scientific_protocol=True,other_methods_modified=False,per_object_tuning=False,
        issue='The export now includes every Background Gaussian with positive native visible contribution in the frozen mask. This includes tails and background structures outside the human; the six-view full-support height can therefore describe the entire contaminated Gaussian package rather than the body.',
        transformation='A = T_world_from_camera @ diag(s,s,s,1) @ T_camera_from_native, where s = annotation_vertical_extent / full_package_native_alpha_vertical_extent. The native physical gauge is one.',
        depth_implication='This scales both geometry and its source-camera translation: p_camera_new = s * p_camera_native. It preserves source projection but moves all points radially about the camera origin. With full-package height inflated by Background, s becomes small and originally metre-calibrated body/support points move metres toward the camera. Far Gaussian supports may also produce metre-scale geometric outliers before this conditioning.',
        interpretation='Both unchanged-protocol and original native-metric panels are retained. The large conditioned CD is an interaction between extraction support and full-package height conditioning; it must not be attributed solely to the model human geometry. Native metric is a diagnostic supplement, not a silent replacement of other methods or published values.',
        arithmetic_check='Independent scipy cdist calculation of symmetric unsquared CD, strict-threshold F-scores and full-reference denominator, plus independent equal-identity aggregate check; no GPU rerender.')
    checked_panels=0
    for method in ['streetgs','omnire','splatad']:
        root=GEOM/method;summary=read(root/'summary.json');rows=[read(p) for p in sorted((root/'results').glob('*.json'))]
        scales=[];heights=[];examples=[];checked_this=0
        for row in rows:
            ref=refs[row['id']];assert sha(ref['queries'])==row['queries_sha256']
            with np.load(ref['queries']) as z:Q=z['q'];o=z['o'];u=z['u']
            for panel,p in row.get('panels',{}).items():
                path=p['returns'] if panel=='native_metric' else row['returns']
                digest=p['returns_sha256'] if panel=='native_metric' else row['returns_sha256']
                assert sha(path)==digest
                with np.load(path) as z:hit=z['hit'];ranges=z['range_pred']
                m=p['metrics'];assert m['reference_count']==len(Q) and m['hit_count']==int(hit.sum())
                assert abs(m['ray_hit']-float(hit.mean()))<1e-12
                if hit.any():
                    P=o[hit]+ranges[hit,None]*u[hit];dpg,dgp=dist_means(P,Q)
                    assert abs(float((dpg.mean()+dgp.mean())/2)-m['cd_m'])<1e-10
                    for threshold in [.05,.1,.2]:
                        pr=float((dpg<threshold).mean());re=float((dgp<threshold).mean())
                        f=2*pr*re/(pr+re) if pr+re else 0
                        assert abs(f-m[f'f1_{round(threshold*100)}cm'])<1e-12
                else:assert m['cd_m'] is None and m['f1_10cm']==0
                checked_this+=1
            if row.get('certificate'):
                c=read(row['certificate']);s=c['scale_m_per_native_unit'];scales.append(s);heights.append(c['native_vertical_extent'])
                examples.append(dict(id=row['id'],reference_panel=row['reference_panel'],scale=s,package_height_m=c['native_vertical_extent'],annotation_height_m=c['annotation_vertical_extent_m'],
                    native_CD_cm=row['panels']['native_metric']['metrics']['cd_m']*100 if row['panels']['native_metric']['metrics']['cd_m'] is not None else None,
                    conditioned_CD_cm=row['panels']['annotation_vertical_extent_conditioned']['metrics']['cd_m']*100 if row['panels']['annotation_vertical_extent_conditioned']['metrics']['cd_m'] is not None else None))
        panels={}
        for panel in ['main','low_support_exploratory']:
            panels[panel]={}
            for measurement in ['native_metric','annotation_vertical_extent_conditioned']:
                values=[r['panels'][measurement]['metrics'] for r in rows if r['reference_panel']==panel and measurement in r.get('panels',{})]
                finite=[m['cd_m'] for m in values if m['cd_m'] is not None]
                cd=float(np.mean(finite)*100) if finite else None;f10=float(np.mean([m['f1_10cm'] for m in values])*100) if values else None
                old=summary['groups'][panel][measurement]
                assert len(finite)==old['CD_cm']['n'] and len(values)==old['F10']['n']
                if cd is not None:assert abs(cd-old['CD_cm']['value'])<1e-10
                if f10 is not None:assert abs(f10-old['F10']['value'])<1e-10
                panels[panel][measurement]=dict(CD_cm=cd,CD_n=len(finite),F10_percent=f10,F10_n=len(values),valid_empty_n=sum(m['cd_m'] is None for m in values))
        result['methods'][method]=dict(status=summary['status'],counts=dict(collections.Counter(r['status'] for r in rows)),panels=panels,
            scale_distribution=quantiles(scales),native_full_support_height_m=quantiles(heights),scale_below_half=sum(s<.5 for s in scales),scale_above_two=sum(s>2 for s in scales),
            package_height_above_5m=sum(v>5 for v in heights),examples_sorted_by_scale=sorted(examples,key=lambda x:x['scale']),
            protocol=str(root/'protocol.json'),protocol_sha256=sha(root/'protocol.json'),summary_sha256=sha(root/'summary.json'),
            independently_verified_metric_panels=checked_this)
        checked_panels+=checked_this
    result['independently_verified_metric_panels']=checked_panels
    result['self_script_sha256']=sha(__file__)
    target=GEOM/'support_scale_diagnostic.json';target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({m:{'status':d['status'],'counts':d['counts'],'main':d['panels']['main'],'scale':d['scale_distribution']} for m,d in result['methods'].items()},indent=2))


if __name__=='__main__':main()
