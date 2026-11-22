"""Additional small-file checks: source relation, extent images and terminal NA evidence."""
from common import *
import numpy as np
from collections import Counter
def main():
    jobs={(r['method'],r['id']):r for r in read(E/'intake.json')['rows']};accepted={(r['method'],r['id']):r for r in read(E/'registration_recovery/accepted.json')['rows']}
    ledger=read(E/'geometry_ledger.json')['rows'];assert len(ledger)==600
    image_count=0;resolution=Counter();layers=Counter();deep=[];max_boundary=0.;max_pitch=0.;image_hashes={}
    for row in ledger:
        job=jobs[row['method'],row['id']];assert row['identity_key']==job['identity_key'] and row['t_ref_us']==job['t_ref_us']
        if row['geometry_status'].startswith('NA'):
            assert sha(row['evidence'])==row['evidence_sha256'];continue
        assert row['geometry_status'] in ['scored','empty_pred']
        d=read(row['result']);c=read(d['certificate']);extent=read(c['extent']);ev=c['evidence'];V=np.asarray(c['T_source_camera_from_native'])
        if row['method'] in ['omnire','splatad']:
            checked(job['source_certificate'],job['source_certificate_sha256']);Twc=np.array(read(job['camera_contract'])['T_world_from_camera_opencv']);world=np.array(ev['preparation']['T_world_from_native'])
            np.testing.assert_allclose(V,np.linalg.inv(Twc)@world,atol=1e-12,rtol=0)
            np.testing.assert_allclose(d['panels']['native_metric']['T_world_from_native'],world,atol=1e-12,rtol=0)
            temporal=ev['preparation']['temporal'];assert ev['preparation']['arrays_at_exact_physical_query']
        else:
            rec=accepted.get((row['method'],row['id']));original=rec or job
            np.testing.assert_allclose(V,original['T_source_camera_from_native'],atol=1e-12,rtol=0)
            assert ev['asset_sha256']==job['asset_sha256'] and ev['asset']==job['asset']
            if ev.get('source_trace'):checked(ev['source_trace'],ev['source_trace_sha256'])
            if rec:
                checked(rec['registration'],rec['registration_sha256']);checked(rec['semantic_review']['evidence'],rec['semantic_review']['evidence_sha256'])
                assert rec['LiDAR_endpoints_used'] is False and len(rec['joints'])>=6
        precision=extent['precision'];resolution[str(precision['fine_resolution'])]+=1;max_boundary=max(max_boundary,precision['boundary_fraction_of_height']);max_pitch=max(max_pitch,precision['fine_pixel_fraction_of_height'])
        for v in extent['views']:
            checked(v['path'],v['sha256']);image_hashes[v['path']]=v['sha256'];image_count+=1
            diag=v.get('depth_diagnostics')
            if row['method']=='trellis2' and diag:
                layers[str(diag['layers_requested'])]+=1
                assert diag['next_layer_max_remaining']<=1e-4
                if diag['layers_requested']>8:deep.append(dict(id=row['id'],kind='extent',view=v['view']['name'],resolution=v['view']['resolution'],**diag))
        if row['method']=='trellis2':
            for diag in d['panels']['annotation_vertical_extent_conditioned']['diagnostics']:
                layers[str(diag['layers_requested'])]+=1;assert diag['next_layer_max_remaining']<=1e-4
                if diag['layers_requested']>8:deep.append(dict(id=row['id'],kind='query',**diag))
    proof=dict(status='passed',terminal_records=600,extent_images_verified=image_count,extent_resolution_counts=resolution,max_boundary_fraction=max_boundary,max_pixel_pitch_fraction=max_pitch,trellis2_all_render_layer_counts=layers,trellis2_deeper_renders=deep,script_sha256=sha(__file__),ledger_sha256=sha(E/'geometry_ledger.json'))
    save(E/'validation/geometry_evidence_audit.json',proof);save(E/'validation/extent_image_hashes.json',image_hashes);print(proof)
if __name__=='__main__':main()
