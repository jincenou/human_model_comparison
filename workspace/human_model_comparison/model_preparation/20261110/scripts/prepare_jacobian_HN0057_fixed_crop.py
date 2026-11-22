"""Prepare a real 30-exposure pilot with one fixed crop and calibrated K.

Jacobian accepts a single K.npy. Moving per-frame crops have changing principal
points, so their image files must not be paired with a single frozen K.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
import yaml

P=Path(__file__).resolve().parents[1]
SOURCE=P/'reports/exavatar_real_sequence_HN0057_20261113/inputs.json'
JA=P/'sources/jacobianavatar'
SCENE='HN0057_native_pilot_20261113'
OUT=P/('reports/jacobian_real_sequence_HN0057_20261113_training')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

frames=json.loads(SOURCE.read_text())['frames']
boxes=np.array([f['crop_xyxy']for f in frames],dtype=np.int64)
crop=np.r_[boxes[:,:2].min(0),boxes[:,2:].max(0)]
all_K=[]
for f in frames:
    k=np.array(f['K_input'],np.float64)
    k[0,2]+=f['crop_xyxy'][0];k[1,2]+=f['crop_xyxy'][1]
    all_K.append(k)
assert np.max(np.abs(np.array(all_K)-all_K[0]))==0
k=all_K[0].copy();k[0,2]-=crop[0];k[1,2]-=crop[1]
assert len(frames)==30 and not OUT.exists()
(OUT/'images').mkdir(parents=True)
records=[]
for f in frames:
    i=int(f['native_index']);src=Path(f['raw_source_path'])
    assert sha(src)==f['raw_source_sha256']
    image=Image.open(src).convert('RGB')
    assert 0<=crop[0]<crop[2]<=image.width and 0<=crop[1]<crop[3]<=image.height
    dest=OUT/'images'/f'{i:05d}.png';image.crop(tuple(crop)).save(dest)
    records.append(dict(index=i,timestamp_us=f['frame_timestamp_micros'],source=str(src),
                        source_sha256=f['raw_source_sha256'],image=str(dest),sha256=sha(dest)))
np.save(OUT/'K.npy',k)
base=JA/'configs/legacy/SMPLX_dataloader/CUSTOM/my_scene.yaml'
scene=yaml.safe_load(base.read_text());scene['Path'].update(training_folder=str(OUT),novel_pose_folder=str(OUT),
    save_root=str(OUT/'outputs'),exavatar_pose_folder='')
config=base.with_name(SCENE+'.yaml');assert not config.exists();config.write_text(yaml.safe_dump(scene,sort_keys=False))
recipe_base=JA/'configs/datasets/Custom-my_scene.yaml'
recipe=yaml.safe_load(recipe_base.read_text());recipe.update(scene_name=SCENE,data_path=str(OUT),save_root_override=str(OUT/'outputs/default'))
recipe_path=recipe_base.with_name('Custom-'+SCENE+'.yaml');assert not recipe_path.exists();recipe_path.write_text(yaml.safe_dump(recipe,sort_keys=False))
manifest=dict(identity='HN0057',status='real_inputs_prepared_preprocessing_not_started',
    real_frames=30,generated_frames=0,interpolated_frames=0,source_manifest=str(SOURCE),source_manifest_sha256=sha(SOURCE),
    crop_policy='One fixed union of the 30 already tracked native crop boxes; no per-frame resize or recenter',
    crop_xyxy=crop.tolist(),K=k.tolist(),K_sha256=sha(OUT/'K.npy'),
    known_prior='nuScenes camera intrinsics and existing identity track for fixed crop selection',
    target_association='First-frame native detector and subsequent SAMURAI tracking still require identity review',
    input_difference_from_exavatar='Same 30 raw exposures; fixed 650x333 crop with calibrated K, not moving 176x320 virtual-camera crops',
    short_clip_limitation='30 real frames; upstream recommends 50–600. No duplicates used to inflate observations',
    reference_timestamp_us=1533201591527893,reference_index=17,
    scene_config=str(config),scene_config_sha256=sha(config),recipe=str(recipe_path),recipe_sha256=sha(recipe_path),
    template_sha256={str(base):sha(base),str(recipe_base):sha(recipe_base)},frames=records)
(OUT/'inputs.json').write_text(json.dumps(manifest,indent=2))
preview=Image.new('RGB',(650,333*3),'white')
for row,i in enumerate([0,15,29]):preview.paste(Image.open(OUT/'images'/f'{i:05d}.png'),(0,row*333))
preview.save(OUT/'input_review.jpg')
print(OUT);print('Fixed crop',crop.tolist(),'K unchanged across real exposures')
