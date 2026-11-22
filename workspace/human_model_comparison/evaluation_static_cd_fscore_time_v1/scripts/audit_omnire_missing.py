from human_asset_benchmark.paths import expand as _hmc_expand
import sys
from common import *
import torch
from omegaconf import OmegaConf
sys.path.insert(0,_hmc_expand('${DRIVESTUDIO_ROOT}'))
from datasets.driving_dataset import DrivingDataset
torch.set_num_threads(2)
item=next(r for r in read(H/'inputs/scene_extract_v2/omnire_plan.json')['objects'] if r['id']=='HN0397');checked(item['config'],item['config_sha256']);cfg=OmegaConf.load(item['config']);data=DrivingDataset(data_cfg=cfg.data)
init_cfg=OmegaConf.to_container(cfg.model.DeformableNodes.init,resolve=True);native=data.get_init_objects(cur_node_type='DeformableNodes',**init_cfg);keys=list(native);true=[int(data.pixel_source.instances_true_id[k]) for k in keys]
state=torch.load(item['checkpoint'],map_location='cpu',weights_only=False)['models']['DeformableNodes'];assert len(keys)==len(state['instances_size'])
for i,k in enumerate(keys):
    torch.testing.assert_close(state['instances_size'][i],native[k]['size'].cpu(),atol=1e-5,rtol=0)
    assert torch.equal(state['instances_fv'][:,i],native[k]['frame_info'].cpu())
target=item['native_true_ids'][0];alltrue=data.pixel_source.instances_true_id.cpu().tolist();indices=[i for i,x in enumerate(alltrue) if x==target];assert len(indices)==1
i=indices[0];fv=data.pixel_source.per_frame_instance_mask[:,i];poses=data.pixel_source.instances_pose[:,i];trans=poses[fv,:3,3];length=float(torch.linalg.vector_norm(trans[1:]-trans[:-1],dim=-1).sum())
save(E/'validation/omnire_missing_reextraction.json',dict(id='HN0397',method='omnire',status='NA_native_actor_filtered_by_original_init' if target not in true else 'potential_recovery',target_true_id=target,identity_key=item['native_id'],original_init_options=init_cfg,native_initialized_true_ids=true,all_checkpoint_sizes_and_visibility_verified=True,visible_training_frames=int(fv.sum()),training_trajectory_length=length,checkpoint_sha256=sha(item['checkpoint']),config_sha256=sha(item['config']),instances_sha256=sha(item['instances_info']),script_sha256=sha(__file__),native_init_sha256=sha(_hmc_expand('${DRIVESTUDIO_ROOT}/datasets/driving_dataset.py')),annotation_assisted_background='Not isolated: background IDs have no target-instance identity; OBB alone insufficient to distinguish static human from nearby scene geometry.'))
print('target',target,'initialized',target in true,'trajectory_length',length,flush=True)
