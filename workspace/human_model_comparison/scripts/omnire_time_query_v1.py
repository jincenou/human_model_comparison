"""Read-only-in-effect evaluation at physical source time; restore all model fields."""
from contextlib import contextmanager
import torch
from models.gaussians.basics import interpolate_quats


@contextmanager
def at_physical_exposure(model, instance, audit):
    assert audit['status']=='mapped' and audit['continuous_bracket'] is not None
    anchor=int(model.cur_frame);assert anchor==audit['source_frame']
    assert not model.in_test_set, 'This adapter is validated for source-protected training exposures'
    bracket=audit['continuous_bracket'];lo,hi=bracket['lo'],bracket['hi'];fraction=bracket['fraction']
    assert 0<=fraction<=1 and 0<=lo<=hi<model.num_frames
    assert bool(model.instances_fv[lo,instance]) and bool(model.instances_fv[hi,instance]), 'Target not represented at both physical-time brackets'
    original_times=model.normalized_timestamps
    original_translation=model.instances_trans[anchor,instance].detach().clone()
    original_quaternion=model.instances_quats[anchor,instance].detach().clone()
    left_t=model.instances_trans[lo,instance].detach().clone();right_t=model.instances_trans[hi,instance].detach().clone()
    left_q=model.instances_quats[lo,instance].detach().clone();right_q=model.instances_quats[hi,instance].detach().clone()
    translation=(1-fraction)*left_t+fraction*right_t
    quaternion=interpolate_quats(left_q[None].clone(),right_q[None].clone(),fraction)[0]
    query=(1-fraction)*original_times[lo]+fraction*original_times[hi]
    assert abs(float(query)-bracket['normalized_query'])<1e-6
    try:
        with torch.no_grad():
            model.normalized_timestamps=original_times.clone();model.normalized_timestamps[anchor]=query
            model.instances_trans[anchor,instance].copy_(translation)
            model.instances_quats[anchor,instance].copy_(quaternion)
        yield dict(policy='piecewise physical sample time to learned normalized time; native continuous deform network plus learned-root linear translation/native quaternion SLERP',
            native_grid_timestamp_us=audit['grid_timestamp_us'],query_timestamp_us=audit['timestamp_us'],
            native_grid_minus_exposure_us=audit['grid_minus_exposure_us'],query_time_error_us=0,
            lo_frame=lo,hi_frame=hi,fraction=fraction,normalized_query=float(query),
            sampled_root_translation=translation.cpu().tolist(),sampled_root_quaternion=quaternion.cpu().tolist(),
            root_translation_change_norm=float(torch.linalg.vector_norm(translation-original_translation)),
            temporal_adapter_variant=True,joint_pose_optimization=False,checkpoint_modified=False)
    finally:
        with torch.no_grad():
            model.normalized_timestamps=original_times
            model.instances_trans[anchor,instance].copy_(original_translation)
            model.instances_quats[anchor,instance].copy_(original_quaternion)
        assert torch.equal(model.instances_trans[anchor,instance],original_translation)
        assert torch.equal(model.instances_quats[anchor,instance],original_quaternion)
