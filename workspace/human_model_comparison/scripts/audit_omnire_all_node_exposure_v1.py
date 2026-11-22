"""Auxiliary full-scene time audit; never rewrites exports or metric inputs."""
import gc
import argparse
from pathlib import Path
import torch
from native_reaudit_omnire_v1 import REPORT,EXPORT,Scene,read,save,sha
from audit_omnire_semantic_timing_v1 import process

OUT=REPORT/'all_node_exposure_aux'
TOLERANCE=dict(target_parameter_max_error=1e-5,
    background_omitted_weight_fraction=1e-5,
    independent_selection_mask_rgb_mae=1e-3,
    independent_selection_mask_rgb_max=1/255,
    independent_selection_mask_fraction_channels_over_1byte=0.01)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'fixed_tolerances.json').exists():
        assert read(OUT/'fixed_tolerances.json')['thresholds']==TOLERANCE
    else:
        save(OUT/'fixed_tolerances.json',dict(thresholds=TOLERANCE,
            decision='Diagnostic flags only. No automatic replacement, exclusion, per-object tuning or metric edits.',
            original_exports_modified=False))
    groups=read(REPORT/'plan.json')['groups'];priority=set(read(REPORT/'semantic_review.json')['confirmed_background_recovery_ids'])
    groups=sorted(enumerate(groups),key=lambda item:not any(o['id'] in priority for o in item[1]['objects']))
    for group_index,group in groups:
        if group_index%args.shards!=args.shard:continue
        todo=[o for o in group['objects'] if not (OUT/o['id']/'audit.json').exists()]
        if not todo:continue
        scene=Scene(group)
        for obj in todo:process(scene,obj,emit_other_previews=False,correct_exposure=True,output=OUT)
        del scene;gc.collect();torch.cuda.empty_cache()
    save(OUT/f'shard_{args.shard}_of_{args.shards}_completed.json',dict(status='completed',shard=args.shard,shards=args.shards))
    if len(list(OUT.glob('*/audit.json')))!=100:return
    rows=[]
    for p in sorted(OUT.glob('*/audit.json')):
        d=read(p);a=d['all_node_time_audit'];sid=d['id'];fraction=a['background_weight_omitted_by_old_selection']/max(a['corrected_background_visible_weight'],1e-30)
        flags=[]
        if max(a['target_gaussian_parameter_max_errors'].values())>TOLERANCE['target_parameter_max_error']:flags.append('target_parameter_changed')
        if fraction>TOLERANCE['background_omitted_weight_fraction']:flags.append('background_omitted_weight_fraction')
        for k in ['independent_selection_mask_rgb_mae','independent_selection_mask_rgb_max','independent_selection_mask_fraction_channels_over_1byte']:
            if a[k]>TOLERANCE[k]:flags.append(k)
        rows.append(dict(id=sid,full_scene_corrected=str(p.parent/'all_node_time_full.png'),
            full_scene_corrected_sha256=sha(p.parent/'all_node_time_full.png'),
            old_full_scene_label='Target at physical exposure; other instance roots at native anchor (diagnostic)',
            corrected_full_scene_label='All bracket-valid instance roots and deformation clocks at physical exposure; unbracketed instances retain native anchor',
            validation=str(EXPORT/sid/'validation.json'),validation_sha256=sha(EXPORT/sid/'validation.json'),
            auxiliary_audit=str(p),auxiliary_audit_sha256=sha(p),
            background_omitted_weight_fraction=fraction,fixed_tolerance_flags=flags,**a))
    assert len(rows)==100,len(rows)
    save(OUT/'summary.json',dict(status='completed',count=len(rows),thresholds=TOLERANCE,
        flagged_ids=[r['id'] for r in rows if r['fixed_tolerance_flags']],
        original_exports_modified=False,formal_metric_inputs_changed=False,rows=rows))

if __name__=='__main__':main()
