"""Record real versus resampled/generated inputs from retained provenance."""
import csv
import json
from pathlib import Path
import yaml
import human_metric_batch_v1 as b
from active_human_results import resolve


def yaml_field(node, *keys):
    # Inspect tagged YAML without instantiating Python objects or constructors.
    for key in keys:
        node = next(v for k,v in node.value if k.value == key)
    return node


def main():
    h = b.H; out = h/'reports/input_budget_audit_v1'; out.mkdir(exist_ok=True)
    summary = b.read(resolve('core_summary'))
    b.verified(summary['selection'], summary['selection_sha256'])
    selected = b.read(summary['selection'])['objects']
    core = {(r['id'],r['method']):r for r in summary['rows']}
    inventory = b.read(h/'reports/common_human_orbit_v1/scene_input_inventory.json')
    mapping = {(r['id'],r['method']):r.get('mapping',{}) for r in inventory['rows']}
    rows, evidence = [], []
    def tracked(path):
        path=Path(path);evidence.append(dict(path=str(path),sha256=b.m.sha(path)))
        return b.read(path)
    for person in sorted(selected,key=lambda r:r['id']):
        sid=person['id']; paths=person['paths']
        native=tracked(paths['native_metadata']); time=tracked(paths['frame_mapping'])
        assert len(native['frames'])==time['native_frames']
        times=[f['frame_timestamp_micros'] for f in native['frames']]
        assert len(times)==len(set(times)) and times==sorted(times)
        folder=Path(paths['production_config']).parents[1]
        generated=tracked(folder/'data'/sid/'fdanyone.json')
        real_sources=len({f['raw_source_sha256'] for f in native['frames']})
        assert real_sources==len(times)
        common=dict(id=sid,scene=person['scene'],reference_timestamp_us=person['t_ref_us'])
        rows.append(dict(**common,method='ours',input_kind='single_camera_real_video_plus_generated_views',
            real_images=real_sources,real_duration_seconds=(times[-1]-times[0])/1e6,
            model_input_slots=len(time['frames']),interpolated_slots=sum(bool(f['interpolated']) for f in time['frames']),
            generated_views=len(generated['cameras']),generated_frames_per_view=generated['frame_count'],
            generated_total_frames=len(generated['cameras'])*generated['frame_count'],
            backend_train_views=len(generated['train_views']),synthetic_reserved_views=len(generated['heldout_views']),
            generated_width=generated['width'],generated_height=generated['height'],
            evidence_scope='Provenance manifests; generated reserved views are not real held-out observations'))
        for method in ['ah','trellis_original','trellis2']:
            item=dict(**common,method=method,input_kind='single_frozen_real_image',real_images=1,
                      real_duration_seconds=0,model_input_slots=1,interpolated_slots=0)
            if method=='ah':
                row=core[sid,method];result=tracked(row['result']);cert=tracked(result['registration_record'])
                trace=tracked(cert['trace_record']); inp=tracked(trace['input_manifest'])
                item['input_manifest']=trace['input_manifest']
                run=Path(cert['native_reload_validation']).parents[1]
                frames=list((run/'outputs'/sid/'multiview').glob('*.png'))
                item.update(generated_views=len(frames) if frames else None,
                    generated_frame_inventory=str(run/'outputs'/sid/'multiview'),
                    evidence_scope='One source exposure; AHC inferred camera and generated multiview assets; retained PNG inventory')
            else:
                item.update(generated_views=0,evidence_scope='Single image-to-3D conditioning; no external generated view set supplied; latent model priors not counted as observations')
            rows.append(item)
        for method in ['omnire','splatad']:
            item=dict(**common,method=method,input_kind='existing_multi_camera_scene_checkpoint',
                      core_status=core[sid,method]['status'],additional_scene_training=False,
                      evidence_scope='Scene input inventory/configuration; not a per-person visibility count or equal-input comparison')
            context=mapping.get((sid,method),{});plan_path=context.get('plan')
            if plan_path:
                plan=tracked(plan_path);matches=[r for r in plan['objects'] if r['id']==sid]
                assert len(matches)==1
                source=matches[0];config=source.get('config')
                item.update(plan=plan_path,checkpoint=source.get('checkpoint'))
                if config and Path(config).exists():
                    if source.get('config_sha256'):b.verified(config,source['config_sha256'])
                    evidence.append(dict(path=config,sha256=b.m.sha(config)))
                    item['config']=config
                    if method=='omnire':
                        cfg=yaml.safe_load(Path(config).read_text());data=cfg['data']
                        item.update(configured_cameras=json.dumps(data['pixel_source']['cameras']),
                                    configured_test_image_stride=data['pixel_source'].get('test_image_stride'),
                                    configured_start=data.get('start_timestep'),configured_end=data.get('end_timestep'))
                        processed=source.get('processed_data')
                        if processed:
                            files=list((Path(processed)/'images').glob('*.jpg'))
                            item['retained_scene_image_files']=len(files)
                            item['retained_scene_time_indices']=len({f.stem.split('_')[0] for f in files})
                            assert data.get('start_timestep')==0 and data.get('end_timestep')==-1
                            assert data['pixel_source'].get('test_image_stride')==0
                            assert set(int(f.stem.split('_')[-1]) for f in files)==set(data['pixel_source']['cameras'])
                            item['real_images']=len(files)
                            item['lidar_scans']=len(list((Path(processed)/'lidar').glob('*.bin')))
                            sample_path=Path(processed).parents[2]/'train_only_nuscenes/v1.0-trainval/sample.json'
                            if sample_path.exists():
                                samples=tracked(sample_path)
                                if len(samples)*len(data['pixel_source']['cameras'])==len(files):
                                    ts=[x['timestamp'] for x in samples]
                                    item['real_duration_seconds']=(max(ts)-min(ts))/1e6
                                    item['real_images']=len(files)
                                    item['evidence_scope']='Configured six-camera processed inventory matched retained nuScenes sample timestamps; scene-level observations, not target-visible count'
                    else:
                        cfg=yaml.compose(Path(config).read_text())
                        parser=yaml_field(cfg,'pipeline','datamanager','dataparser')
                        split_node=yaml_field(parser,'split_manifest')
                        split=Path(*[x.value for x in split_node.value])
                        b.verified(split,yaml_field(parser,'split_sha256').value)
                        spec=tracked(split); entries=[x for x in spec['files'] if x['split']=='train']
                        cameras=[x.value for x in yaml_field(parser,'cameras').value]
                        lidars=[x.value for x in yaml_field(parser,'lidars').value]
                        images=[x for x in entries if x['channel'].startswith('CAM_') and ('all' in cameras or x['channel'].removeprefix('CAM_') in cameras)]
                        scans=[x for x in entries if x['channel'] in lidars or ('all' in lidars and x['channel'].startswith('LIDAR'))]
                        assert images
                        ts=[x['timestamp_us'] if 'timestamp_us' in x else int(Path(x['filename']).stem.rsplit('__',1)[1]) for x in images]
                        item.update(configured_cameras=json.dumps(cameras),real_images=len(images),
                            real_duration_seconds=(max(ts)-min(ts))/1e6,lidar_scans=len(scans),
                            native_split_manifest=str(split),native_split_sha256=b.m.sha(split),
                            configured_start=yaml_field(parser,'dataset_start_fraction').value,
                            configured_end=yaml_field(parser,'dataset_end_fraction').value,
                            evidence_scope='Hashed native benchmark train-split records selected by configured sensors; scene-level observations, not target-visible count')
                        assert item['configured_start']=='0.0' and item['configured_end']=='1.0'
                else:item['budget_limitation']='No surviving config in mapped extraction plan; native availability is reported separately'
            else:item['budget_limitation']='No mapped independent scene object; no scene-input count inferred'
            rows.append(item)
    assert len(rows)==600 and len({(r['id'],r['method'])for r in rows})==600
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'per_identity_method.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    b.save(out/'audit.json',dict(selection_sha256=summary['selection_sha256'],rows=rows,
        evidence=list({e['path']:e for e in evidence}.values()),equal_input_budget=False,
        limitations=['Retained scene image inventory is not the number of target-visible frames or exact per-iteration training samples.',
                     'Scene counts derive from retained native configuration and split manifests, not a fresh checkpoint-training replay.',
                     'Masks revised for evaluation do not retroactively change generation inputs.']))
    own=[r for r in rows if r['method']=='ours']; ah=[r for r in rows if r['method']=='ah']
    print(json.dumps(dict(identities=100,rows=600,ours_real_images=sorted({r['real_images']for r in own}),
        ours_input_slots=sorted({r['model_input_slots']for r in own}),ours_interpolated_slots=sorted({r['interpolated_slots']for r in own}),
        generated_views=sorted({r['generated_views']for r in own}),generated_frames=sorted({r['generated_frames_per_view']for r in own}),
        ah_retained_views=sorted({str(r['generated_views'])for r in ah})),ensure_ascii=False))


if __name__=='__main__':main()
