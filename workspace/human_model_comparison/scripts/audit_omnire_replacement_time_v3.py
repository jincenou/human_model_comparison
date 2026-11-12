"""Map frozen real exposures to trained scene blocks and continuous time brackets."""
import bisect
import hashlib
import json
from pathlib import Path
import statistics

from omegaconf import OmegaConf

H=Path(__file__).resolve().parents[1]
R=H.parent/'vehicle_model_comparison'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection=H/'inputs/benchmark_human100_v3/selection.json';rows=[r for r in json.loads(selection.read_text())['objects'] if r['id']=='HN0646']
    plan={r['id']:r for r in json.loads((H/'inputs/scene_extract_replacement_v3/omnire_plan.json').read_text())['objects']}
    cache={};records=[]
    for row in rows:
        rec=dict(id=row['id'],cohort=row['cohort'],timestamp_us=row['t_ref_us'],state_time_valid=False)
        try:
            if row['cohort']=='original75':
                v=json.loads((R/'exports/omnire_human95/objects'/row['id']/'validation.json').read_text())
                config=Path(v['checkpoint']).parent/'config.yaml';processed_source=Path(v['processed_source_image'])
            else:
                p=plan[row['id']];config=Path(p['config']);processed_source=Path(p['source_matches'][0])
            cfg=OmegaConf.load(config);data=Path(cfg.data.data_root)/f'{int(cfg.data.scene_idx):03d}'
            base=Path(cfg.data.data_root).parent.parent
            metadata=base/'train_only_nuscenes/v1.0-trainval'
            if not metadata.is_dir() and (base/'input_audit.json').is_file():
                prepared=json.loads((base/'input_audit.json').read_text())
                assert prepared['scene']==row['scene'] and prepared['interpolate_N']==0
                assert Path(prepared['processed_path']).resolve()==data.resolve()
                metadata=Path(prepared['data_root'])/'v1.0-trainval'
            assert metadata.is_dir(),f'Metadata layout requires explicit audit: {metadata}'
            key=str(metadata)
            if key not in cache:
                samples={r['token']:r for r in json.loads((metadata/'sample.json').read_text())}
                sensors={r['token']:r for r in json.loads((metadata/'sample_data.json').read_text())}
                scenes=json.loads((metadata/'scene.json').read_text());assert len(scenes)==1
                chain=[];token=scenes[0]['first_sample_token'];visited=set()
                while token:
                    assert token not in visited;visited.add(token);sample=samples[token];chain.append(sample);token=sample['next']
                assert len(chain)==scenes[0]['nbr_samples']
                assert len(chain)==len(list((data/'images').glob('*_0.jpg')))
                cache[key]=(chain,sensors,sha(metadata/'sample.json'),sha(metadata/'sample_data.json'))
            chain,sensors,sample_sha,sensor_sha=cache[key]
            frame,cam=map(int,processed_source.stem.split('_'))
            exposure=sensors[row['reference_frame']['sample_data_token']]
            assert exposure['timestamp']==row['t_ref_us']
            assert chain[frame]['token']==exposure['sample_token']
            assert sha(processed_source)==row['reference_frame']['raw_source_sha256']
            times=[r['timestamp'] for r in chain];assert all(a<b for a,b in zip(times,times[1:]))
            index=bisect.bisect_left(times,row['t_ref_us'])
            bracket=None
            if index<len(times) and times[index]==row['t_ref_us']:bracket=(index,index,0.)
            elif 0<index<len(times):
                lo,hi=index-1,index;bracket=(lo,hi,(row['t_ref_us']-times[lo])/(times[hi]-times[lo]))
            rec.update(status='mapped',source_image=str(processed_source),source_frame=frame,camera_index=cam,
                metadata_root=str(metadata),sample_table_sha256=sample_sha,sample_data_table_sha256=sensor_sha,
                config_path=str(config),config_sha256=sha(config),num_model_frames=len(times),
                source_sample_token=exposure['sample_token'],grid_timestamp_us=times[frame],
                grid_minus_exposure_us=times[frame]-row['t_ref_us'],native_normalized_time=frame/(len(times)-1),
                model_time_basis='uniform normalized processed-block index; physical sample intervals may differ',
                continuous_bracket=None if bracket is None else dict(lo=bracket[0],hi=bracket[1],fraction=bracket[2],
                    lo_timestamp_us=times[bracket[0]],hi_timestamp_us=times[bracket[1]],
                    normalized_query=(bracket[0]+bracket[2]*(bracket[1]-bracket[0]))/(len(times)-1)),
                native_nearest_frame_is_exact_physical_time=times[frame]==row['t_ref_us'],
                next='Validate continuous deformation query plus learned root pose interpolation at frozen physical exposure; no extrapolation or best-frame choice')
        except Exception as error:rec.update(status='needs_layout_or_mapping_audit',reason=repr(error))
        records.append(rec)
    good=[r for r in records if r['status']=='mapped'];offsets=[abs(r['grid_minus_exposure_us']) for r in good]
    report=dict(selection_sha256=sha(selection),mapped=len(good),total=len(rows),
        abs_offset_us=dict(min=min(offsets),median=statistics.median(offsets),max=max(offsets)) if offsets else None,
        nonzero_offsets=sum(r['grid_minus_exposure_us']!=0 for r in good),
        bracketed=sum(r['continuous_bracket'] is not None for r in good),records=records)
    out=H/'reports/model_time_replacement_v3';out.mkdir(exist_ok=True)
    (out/'omnire.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))
    print([(r['id'],r.get('reason')) for r in records if r['status']!='mapped'])


if __name__=='__main__':main()
