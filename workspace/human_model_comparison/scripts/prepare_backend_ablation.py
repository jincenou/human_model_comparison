"""Assemble exact restored frontend inputs without modifying production assets."""
import argparse
import copy
import os
from pathlib import Path
from common_human_orbit import H,read,save,sha,checked,inputs

SOURCE=H.parent/'asset_work/human_nuscenes100_20261026'
CODE=H.parent/'human'


def prepare(sid):
    active,selected,reference,_,_,_=inputs(sid,'ours')
    cfg_path=SOURCE/'config'/f'{sid}.json';config=read(cfg_path)
    original_manifest=Path(config['provenance']['manifest']);checked(original_manifest,config['provenance']['manifest_sha256']);manifest=read(original_manifest)
    recovered={};receipts=[]
    for path in sorted((H/'runs/backend_ablation_cache_restore'/sid).glob('*/recovery.json')):
        record=read(path)
        if record['status']!='exact_original_rgba_restored':continue
        assert record['manifest_sha256']==sha(original_manifest)and record['diffusion_generation_rerun']is False
        receipts.append(dict(path=str(path),sha256=sha(path)))
        for row in record['frames']:
            key=(row.get('view',record.get('view')),row['frame'])
            checked(row['path'],row['original_sha256']);assert row['sha256']==row['original_sha256']
            if key in recovered:assert recovered[key]['sha256']==row['sha256']
            recovered[key]=row
    expected={(r['view'],r['frame']):r['sha256']for r in manifest['input_hashes']}
    assert set(recovered)==set(expected),f'Exact cache incomplete: {len(recovered)}/{len(expected)}'
    assert all(recovered[k]['sha256']==digest for k,digest in expected.items())
    root=H/'runs/backend_ablation_v1'/sid
    assert not root.exists(),'Inspect existing ablation root; no automatic overwrite'
    data=root/'data'/sid;data.mkdir(parents=True)
    for (view,frame),row in recovered.items():
        path=data/'rgba/images'/f'cam{view:02d}'/f'{frame:06d}.png';path.parent.mkdir(parents=True,exist_ok=True);os.link(row['path'],path)
    canonical=original_manifest.parent/'canonical.npz';checked(canonical,config['provenance']['initialization_sha256']);os.link(canonical,data/'canonical.npz')
    relocated=copy.deepcopy(manifest);relocated['source_data']=str(data/'rgba');save(data/'fdanyone.json',relocated)
    assert {k:v for k,v in relocated.items()if k!='source_data'}=={k:v for k,v in manifest.items()if k!='source_data'}
    updated=copy.deepcopy(config);updated['dataset']['source_path']=str(data)
    updated['provenance'].update(manifest=str(data/'fdanyone.json'),manifest_sha256=sha(data/'fdanyone.json'),initialization=str(data/'canonical.npz'))
    (root/'config').mkdir();save(root/'config'/f'{sid}.json',updated)
    original_cases=read(SOURCE/'config/cases.json');case=original_cases[sid]
    override=SOURCE/'config/case_overrides'/f'{sid}.json'
    if override.exists():
        import hashlib,json
        over=read(override);assert over['original_spec_sha256']==hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest();case=over['specification']
    assert case['prepared_frames']==manifest['frame_count']
    case=dict(case,manifest=str(data/'fdanyone.json'),config=str(root/'config'/f'{sid}.json'))
    save(root/'config/cases.json',{sid:case})
    dependencies=[CODE/'common.py',CODE/'data.py',CODE/'reconstruction.py',*sorted((CODE/'vendor/grid4d').rglob('*.py'))]
    save(root/'protocol.json',dict(status='inputs_frozen_training_not_started',id=sid,identity_key=selected['identity_key'],timestamp_us=selected['t_ref_us'],selection_sha256=active['selection_sha256'],reference=reference,
        original_config=str(cfg_path),original_config_sha256=sha(cfg_path),original_manifest=str(original_manifest),original_manifest_sha256=sha(original_manifest),restored_manifest=str(data/'fdanyone.json'),restored_manifest_sha256=sha(data/'fdanyone.json'),cache_receipts=receipts,rgba_count=len(expected),all_original_rgba_file_hashes_equal=True,
        original_initialization=str(canonical),initialization_sha256=sha(canonical),path_changes_only=['dataset.source_path','provenance.manifest','provenance.manifest_sha256','provenance.initialization','manifest.source_data','case.manifest','case.config'],
        seed=config['seed'],frames=manifest['frame_count'],resolution=[manifest['width'],manifest['height']],sh_degree=config['dataset']['sh_degree'],train_views=manifest['train_views'],generated_heldout_views=manifest['heldout_views'],diffusion_generation_rerun=False,
        variants=[dict(name='native_33k',actual_updates=32999,description='Original native Grid4D recipe, same seed and exact historical frontend inputs'),dict(name='complete_A_53k',actual_updates=52999,description='Original optimized route: native32999 + alpha15000 + lowerLR5000; retain original full budget'),dict(name='rgb_continuation_53k',actual_updates=52999,description='Matched continuation budget and LR/sampling schedule, alpha loss weight zero; explicitly a budget control, not a new original recipe')],
        interpretation='Native33k versus optimized53k includes extra20000 updates; the matched continuation distinguishes alpha-loss effect from extra staged optimization',
        code_hashes={str(p):sha(p)for p in dependencies},source_scoring_protocol='Existing frozen real source-camera core metrics; same physical time, K/T and reference ROI',evaluation_scope='Observed reconstruction; generated-view holdouts are not real unseen-camera ground truth',native_assets_overwritten=False,full_ablation_completed=False))
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);a=p.parse_args();prepare(a.human_id)
