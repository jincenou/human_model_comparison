"""Compact reference/asset inventories and measured resource/conditioning diagnostics."""
import re
import numpy as np
from common import *
def main():
    refs=read(E/'reference_manifest.json')['rows'];jobs=read(E/'intake.json')['rows'];ledger=read(E/'geometry_ledger.json')['rows'];by={(r['method'],r['id']):r for r in ledger}
    fields=['id','identity_key','t_ref_us','scan_timestamp_us','delta_ms','query_count','candidate_panel','reference_status','annotation_vertical_extent_m','scan_token','queries','queries_sha256','scan_sha256','label_sha256']
    csv_write(E/'tables/reference_support.csv',[{k:r.get(k) for k in fields} for r in refs])
    fields=['id','method','identity_key','t_ref_us','asset_available','asset','asset_sha256','source_trace','source_trace_sha256','source_certificate','source_certificate_sha256','prior_2d_status']
    csv_write(E/'tables/asset_inventory.csv',[dict({k:r.get(k) for k in fields},geometry_final_status=by[r['method'],r['id']]['geometry_status'],had_source_relation=bool(r.get('T_source_camera_from_native'))) for r in jobs])
    gaps=[r for r in jobs if r['method'] in ['trellis_original','trellis2'] and r['asset_available'] and not r.get('T_source_camera_from_native')];accepted={(r['method'],r['id']):r for r in read(E/'registration_recovery/accepted.json')['rows']};local=read(E/'registration_recovery/local_views_summary.json')['rows'];rows=[]
    for r in gaps:
        m,sid=r['method'],r['id'];a=accepted.get((m,sid));rows.append(dict(method=m,id=sid,original_correspondences=len(read(E/'registration_recovery/existing'/m/sid/'registration.json')['joints']),zoom_correspondences=len(read(E/'registration_recovery/zoom'/m/sid/'registration.json')['joints']),maximum_local_correspondences=max(len(x['joints']) for x in local if x['method']==m and x['id']==sid),accepted=bool(a),accepted_correspondences=len(a['joints']) if a else None,geometry_final_status=by[m,sid]['geometry_status'],scope='one common correction and direct Codex semantic review; no score-selected transform'))
    csv_write(E/'tables/registration_recovery.csv',rows)
    csv_write(E/'tables/scene_missing_audit.csv',[dict(method=r['method'],id=r['id'],status=r['status'],reason=r['reason'],evidence=str(E/'validation/missing_scene_audit_final.json')) for r in read(E/'validation/missing_scene_audit_final.json')['records']])
    shifts=[]
    for m in ['omnire','splatad']:
        for p in (E/'results'/m).glob('*.json'):
            d=read(p);c=read(d['certificate']);A=np.array(c['T_world_from_native']);W=np.array(d['panels']['native_metric']['T_world_from_native'])
            shifts.append(dict(method=m,id=d['id'],reference_panel=d['reference_panel'],scale_m_per_native_unit=c['scale_m_per_native_unit'],asset_origin_shift_m=float(np.linalg.norm(A[:3,3]-W[:3,3])),conditioned_cd_m=d['panels']['annotation_vertical_extent_conditioned']['metrics']['cd_m'],native_cd_m=d['panels']['native_metric']['metrics']['cd_m']))
    csv_write(E/'tables/scene_conditioning_diagnostics.csv',shifts)
    rs=[__import__('json').loads(s) for s in (E/'timing/resource_samples.jsonl').read_text().splitlines()];summary=dict(samples=len(rs),cpu_logical_count=rs[0]['cpu_count'],load1_max=max(x['load_average'][0] for x in rs),minimum_free_gib=min(x['free_bytes'] for x in rs)/2**30,full_io_pressure_avg10_max_percent=max(float(re.search(r'full avg10=(\S+)',x['io_pressure'])[1]) for x in rs),cpu_pressure_avg10_max_percent=max(float(re.search(r'some avg10=(\S+)',x['cpu_pressure'])[1]) for x in rs),scope='30-second snapshots from late first Ours preprocessing through serial generation, plus resumed export; initial stages and baseline runs have before/after records; no claim of continuous full-process tracing',resource_file_sha256=sha(E/'timing/resource_samples.jsonl'),review='No obvious sustained host saturation in sampled windows: 192 logical CPUs, max1-minute load13.06, CPU pressure<=0.09%, full I/O pressure<=1.92%; GPU4 held exclusive shared+auxiliary leases; native methods executed serially. Short unobserved contention cannot be ruled out.')
    save(E/'validation/time_resource_review.json',summary);print(summary)
if __name__=='__main__':main()
