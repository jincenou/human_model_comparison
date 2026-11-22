"""Use the same 30 native real crops as ExAvatar; no fitted output is reused."""
from pathlib import Path
import hashlib,json,shutil

P=Path(__file__).resolve().parents[1]
SOURCE=P/'reports/exavatar_real_sequence_HN0057_20261113/inputs.json'
OUT=P/'reports/flexi_real_sequence_HN0057_20261113'
NATIVE=P/'sources/flexiavatar/smplx_fitting/data/Custom/data/HN0057_native_pilot_20261113'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
source=json.loads(SOURCE.read_text());frames=source['frames']
assert len(frames)==30 and len({f['frame_timestamp_micros']for f in frames})==30
assert not OUT.exists()and not NATIVE.exists()and not NATIVE.is_symlink()
(OUT/'frames').mkdir(parents=True)
rows=[]
for index,f in enumerate(frames):
    assert f['native_index']==index and not f['interpolated']and not f['enhanced']
    assert sha(f['path'])==f['sha256']and sha(f['raw_source_path'])==f['raw_source_sha256']
    dst=OUT/'frames'/f'{index}.png';shutil.copyfile(f['path'],dst);assert sha(dst)==f['sha256']
    rows.append(dict(index=index,path=str(dst),sha256=sha(dst),raw_source=f['raw_source_path'],
                     raw_source_sha256=f['raw_source_sha256'],timestamp_us=f['frame_timestamp_micros'],
                     native_crop_xyxy=f['crop_xyxy'],K_input=f['K_input']))
indices=''.join(f'{i}\n'for i in range(30))
for name in ['frame_list_all.txt','frame_list_train.txt']:(OUT/name).write_text(indices)
manifest=dict(identity='HN0057',status='real_inputs_prepared_fitting_not_started',real_frames=30,
              generated_frames=0,interpolated_frames=0,real_span_seconds=(rows[-1]['timestamp_us']-rows[0]['timestamp_us'])/1e6,
              input_policy='Exact native real crops also supplied to ExAvatar, copied unchanged; no video re-encoding or duplicate frames',
              upstream_interface='Provide the frames and index files normally produced by extract_frames.py; proceed with tools/run.py',
              reused_fitted_parameters=False,reused_masks=False,
              camera_policy='Preserve calibrated per-frame input K as provenance; native virtual cameras will be produced by FlexiAvatar. Common-camera validation remains pending.',
              variant_scope='Real-only pilot inputs; full MimicMotion auxiliary-view variant remains a separate unfinished task',
              evaluation_scope='Observed-source demo, not a held-out test',reference_index=17,reference_timestamp_us=1533201591527893,
              source_manifest=str(SOURCE),source_manifest_sha256=sha(SOURCE),frames=rows)
(OUT/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
NATIVE.parent.mkdir(parents=True,exist_ok=True);NATIVE.symlink_to(OUT,target_is_directory=True)
print(OUT)
