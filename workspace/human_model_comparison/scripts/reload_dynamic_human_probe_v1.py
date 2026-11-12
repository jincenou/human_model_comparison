"""Fresh-process verification of the small dynamic geometry exports."""
import hashlib
import json
import torch
from types import SimpleNamespace
from audit_dynamic_human_feasibility_v1 import OUT, D, S, sha, native_defs, rigid_residual

torch.set_num_threads(4)
report=json.loads((OUT/'native_probe.json').read_text())
results={}
for method in ['omnire','splatad']:
    row=report[method]
    assert sha(row['bundle'])==row['bundle_sha256']
    bundle=torch.load(row['bundle'],map_location='cpu',weights_only=True)
    frames=row['native_frames'];clouds=[]
    if method=='omnire':
        assert sha(D/'models/modules.py')==row['native_modules_sha256']
        assert sha(D/'models/nodes/deformable.py')==row['native_deformation_sha256']
        definitions=native_defs(D/'models/modules.py',['get_embedder','Embedder','ConditionalDeformNetwork'])
        net=definitions['ConditionalDeformNetwork'](input_ch=3,**bundle['network_config']).eval()
        net.load_state_dict(bundle['network_state'],strict=True)
        means=bundle['canonical']['_means']
        obj=SimpleNamespace(point_ids=torch.zeros((len(means),1),dtype=torch.long),
            instances_embedding=bundle['embedding'],instances_size=bundle['size'],
            normalized_timestamps=bundle['normalized_timestamps'],deform_network=net)
        deform=native_defs(D/'models/nodes/deformable.py',['get_deformation'],'DeformableNodes')['get_deformation']
        with torch.inference_mode():
            for frame in frames:
                obj.cur_frame=frame;clouds.append(means+deform(obj,means)[0])
    else:
        rotate=native_defs(S/'nerfstudio/cameras/camera_utils.py',['rotation_6d_to_matrix'])['rotation_6d_to_matrix']
        r=rotate(bundle['dynamic']['rotations_6d'][frames]);t=bundle['dynamic']['positions'][frames]
        clouds=[bundle['gauss_params']['means']@rr.T+tt for rr,tt in zip(r,t)]
    hashes=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in clouds]
    assert hashes==row['geometry_sha256']
    results[method]=dict(status='fresh_process_geometry_exact',frames=len(frames),all_geometry_hashes_equal=True,
        no_full_scene_checkpoint_loaded=True,rgb_render_validated=False)

# Verify that the diagnostic rejects rigid motion and detects nonrigid change.
torch.manual_seed(42);a=torch.randn(128,3,dtype=torch.float64)
q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q)
b=a@q+torch.tensor([7.,-3.,2.]);assert rigid_residual(a,b)<1e-12
b[:10,0]+=.2;assert rigid_residual(a,b)>.01
results['diagnostic_tests']={'rigid_transform_residual':'passed','nonrigid_perturbation':'passed'}
(OUT/'independent_reload.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
