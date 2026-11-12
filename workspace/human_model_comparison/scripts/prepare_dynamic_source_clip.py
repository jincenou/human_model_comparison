"""Prepare true-exposure cameras/masks and render Ours on the same source clock.

One reusable entry for a small dynamic pilot. Existing 61-frame interpolation
inputs are never treated as real video ground truth. Output masks require review.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,bisect,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp
from prepare_human_source_cameras_v1 import sha,transform
H=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def selected(sid):
 active=read(H/'inputs/active_human_benchmark.json');assert sha(active['selection'])==active['selection_sha256']
 return active,next(r for r in read(active['selection'])['objects'] if r['id']==sid)
def prepare(sid,out):
 import torch
 active,row=selected(sid);native=read(row['paths']['native_metadata']);mapping=read(row['paths']['frame_mapping']);base=Path(row['paths']['native_metadata']).parent
 assert native['track_id']==row['identity_key'].split(':')[-1]
 audit=next(r for r in read(H/'reports/model_time_audit_v1/omnire.json')['records'] if r['id']==sid)
 metadata=Path(audit['metadata_root']);tables={k:metadata/(k+'.json') for k in ['sample_data','ego_pose','calibrated_sensor','sample']}
 records={k:{r['token']:r for r in read(p)} for k,p in tables.items()}
 chain=[];scenes=read(metadata/'scene.json');token=scenes[0]['first_sample_token']
 while token: item=records['sample'][token];chain.append(item);token=item['next']
 model_times=[r['timestamp'] for r in chain]
 omni_path=H/'exports/dynamic_human_feasibility_v1'/f'omnire_{sid}_native_dynamic.pt'
 splat_path=H/'exports/dynamic_human_feasibility_v1'/f'splatad_{sid}_native_dynamic.pt'
 omni=torch.load(omni_path,weights_only=True,map_location='cpu');splat=torch.load(splat_path,weights_only=True,map_location='cpu')
 splat_path_validation=H/'exports/common_camera_human_v1/splatad'/sid/'validation.json'
 if not splat_path_validation.exists():splat_path_validation=H/'exports/common_camera_human_original75_time_v1/splatad'/sid/'validation.json'
 splat_val=read(splat_path_validation);offset=splat_val['common_camera_audit']['native_time_offset_seconds']
 splat_times=splat['dynamic']['timestamps'].numpy()
 cp=base/'native_cameras.npz';assert sha(cp)==native['camera_npz_sha256'];cams=np.load(cp)
 if out.exists():assert not any(out.iterdir()),'Only an empty failed preparation folder can be reused'
 out.mkdir(parents=True,exist_ok=True);frames=[];excluded=[];anchor=read(H/'configs/human_source_common_v1/reviewed_source_v2'/sid/'camera.json')
 for i,f in enumerate(native['frames']):
  r=records['sample_data'][f['sample_data_token']];t=f['frame_timestamp_micros'];assert r['timestamp']==t and not f['interpolated'] and not f['enhanced']
  raw=Path(_hmc_expand('${NUSCENES_ROOT}'))/r['filename'];assert raw==Path(f['raw_source_path']) and sha(raw)==f['raw_source_sha256']
  sensor=records['calibrated_sensor'][r['calibrated_sensor_token']];ego=records['ego_pose'][r['ego_pose_token']]
  K=np.array(sensor['camera_intrinsic']);T=transform(ego)@transform(sensor);A=np.array(f['undistorted_to_input'])
  np.testing.assert_allclose(A@K,f['K_input'],atol=1e-8,rtol=0)
  np.testing.assert_allclose(T,cams['image_center_cv_camera_to_world'][i],atol=1e-8,rtol=0)
  if t==row['t_ref_us']:
   np.testing.assert_allclose(T,anchor['T_world_from_camera_opencv'],atol=1e-8,rtol=0);np.testing.assert_allclose(K,anchor['K_raw'],atol=1e-8,rtol=0)
  elapsed=(t-native['frames'][0]['frame_timestamp_micros'])/1e6
  exact=[x['slot'] for x in mapping['frames'] if not x['interpolated'] and abs(x['source_relative_exposure_seconds']-elapsed)<1e-8]
  sq=t/1e6-offset
  missing=[]
  if not model_times[0]<=t<=model_times[-1]:missing.append('omnire')
  if not float(splat_times[0])<=sq<=float(splat_times[-1]):missing.append('splatad')
  if missing:
   excluded.append(dict(index=i,timestamp_us=t,time_seconds=elapsed,missing_methods=missing,
    reason='outside_native_scene_time_support',source_rgb=str(raw),source_rgb_sha256=sha(raw)))
   continue
  assert not excluded,'Only an unsupported tail is allowed; never rebase or bridge a missing interval'
  hi=bisect.bisect_left(model_times,t);assert 0<hi<len(model_times)
  lo=hi if model_times[hi]==t else hi-1;fraction=0 if lo==hi else (t-model_times[lo])/(model_times[hi]-model_times[lo])
  shi=int(np.searchsorted(splat_times,sq));slo=shi-1
  assert 0<shi<len(splat_times) and omni['visibility'][lo] and omni['visibility'][hi] and splat['dynamic']['visibility'][slo] and splat['dynamic']['visibility'][shi]
  crop=f['crop_xyxy'];bbox=np.array(f['bbox_input_xyxy'])+np.array([crop[0],crop[1],crop[0],crop[1]])
  frames.append(dict(index=i,timestamp_us=t,time_seconds=elapsed,sample_data_token=r['token'],source_rgb=str(raw),source_rgb_sha256=sha(raw),
   K_raw=K.tolist(),T_world_from_camera_opencv=T.tolist(),render_wh=[r['width'],r['height']],prompt_bbox_xyxy=bbox.tolist(),
   native_crop_xyxy=crop,exact_legacy_mask_slots=exact,omnire=dict(lo=lo,hi=hi,fraction=fraction,normalized_query=(lo+fraction*(hi-lo))/(len(chain)-1)),
   splatad=dict(query_seconds=sq,lo=slo,hi=shi,fraction=float((sq-splat_times[slo])/(splat_times[shi]-splat_times[slo])))))
 assert len(frames)>=2 and frames[0]['time_seconds']==0
 extra={}
 if excluded:
  assert any(f['timestamp_us']==row['t_ref_us'] for f in frames),'Frozen reference must remain supported'
  record=dict(id=sid,status='shared_native_scene_time_interval_selected_before_scores',
   full_capture=str(row['paths']['native_metadata']),full_capture_sha256=sha(row['paths']['native_metadata']),
   source_real_frame_count=len(native['frames']),common_real_frame_count=len(frames),
   source_pair_count=len(native['frames'])-1,common_pair_count=len(frames)-1,
   excluded_real_exposures=excluded,omnire_supported_interval_us=[model_times[0],model_times[-1]],
   splatad_supported_interval_seconds=[float(splat_times[0]),float(splat_times[-1])],
   rule='All real exposures within native time support; no quality or metric selection',
   time_rebased=False,loop_hold_or_interpolated_reference_used=False,full_capture_preserved=True,
   script_sha256=sha(__file__))
  save(out/'scene_temporal_support.json',record)
  extra=dict(source_real_frame_count=len(native['frames']),temporal_support=str(out/'scene_temporal_support.json'),
   temporal_support_sha256=sha(out/'scene_temporal_support.json'))
 save(out/'clip.json',dict(id=sid,identity_key=row['identity_key'],selection_sha256=active['selection_sha256'],status='real_exposure_cameras_and_model_time_intersection_ready',
  frames=frames,real_frame_count=len(frames),duration_seconds=frames[-1]['time_seconds'],reference_frame_index=row['reference_frame']['native_index'],
  model_input_slots=len(mapping['frames']),legacy_mask_exact_real_frames=sum(bool(f['exact_legacy_mask_slots']) for f in frames),
  metadata_table_sha256={k:sha(p) for k,p in tables.items()},native_metadata_sha256=sha(row['paths']['native_metadata']),native_cameras_sha256=sha(cp),
  metadata_source=str(metadata),camera_npz_global_pose_matches_metadata=True,anchor_matches_accepted_static_camera=True,
  dynamic_bundles={m:dict(path=str(p),sha256=sha(p)) for m,p in [('omnire',omni_path),('splatad',splat_path)]},
  reference_masks_ready=False,temporal_metric_eligible=False,source_reconstruction_only=True,script_sha256=sha(__file__),**extra))
 print(json.dumps(dict(frames=len(frames),real_duration=frames[-1]['time_seconds'],legacy_exact_masks=sum(bool(f['exact_legacy_mask_slots']) for f in frames))),flush=True)

def masks(sid,out):
 import torch
 from prepare_exact_reference_masks_v2 import SOURCE,WEIGHT,EXPECTED_WEIGHT_SHA,CONFIG
 sys.path.insert(0,str(SOURCE));from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 assert sha(WEIGHT)==EXPECTED_WEIGHT_SHA;torch.set_num_threads(4)
 predictor=SAM2ImagePredictor(build_sam2(CONFIG,str(WEIGHT),device='cuda'))
 clip=read(out/'clip.json');folder=out/'reference_masks';folder.mkdir(exist_ok=False);images=[];rows=[]
 with torch.inference_mode():
  for f in clip['frames']:
   rgb=np.asarray(Image.open(f['source_rgb']).convert('RGB'));assert sha(f['source_rgb'])==f['source_rgb_sha256']
   predictor.set_image(rgb);mask,score,_=predictor.predict(box=np.array(f['prompt_bbox_xyxy'],dtype=np.float32),multimask_output=False)
   a=mask[0].astype(np.uint8)*255;p=folder/f"{f['index']:05d}.png";Image.fromarray(a).save(p)
   rows.append(dict(index=f['index'],timestamp_us=f['timestamp_us'],path=str(p),sha256=sha(p),sam_score=float(score[0])))
   white=np.where((a>127)[...,None],rgb,255).astype(np.uint8);crop=Image.fromarray(white).crop(f['native_crop_xyxy']).resize((132,240))
   tile=Image.new('RGB',(132,264),'white');tile.paste(crop,(0,24));ImageDraw.Draw(tile).text((3,4),f"{f['index']:02d} {f['time_seconds']:.3f}s",fill='black');images.append(tile)
   print('mask',f['index'],flush=True)
 grid=Image.new('RGB',(132*10,264*math.ceil(len(images)/10)),(210,210,210))
 for i,img in enumerate(images):grid.paste(img,((i%10)*132,(i//10)*264))
 grid.save(out/'mask_review.jpg')
 save(out/'mask_candidates.json',dict(status='exact_real_exposure_masks_pending_visual_review',frames=rows,weight_sha256=sha(WEIGHT),method='SAM2.1 base-plus exact-exposure box; no interpolation',
   clip_sha256=sha(out/'clip.json'),script_sha256=sha(__file__),reference_eligible=False))


def support(sid,out):
 """Choose the common native time interval before scoring, preserving full capture."""
 active,row=selected(sid);clip=read(out/'clip.json')
 assert clip['selection_sha256']==active['selection_sha256'] and clip['id']==sid
 assert not (out/'metrics/metrics.json').exists() and not (out/'reference_qualification.json').exists()
 assert not any((out/m/'validation.json').exists() for m in ['ours','omnire','splatad'])
 root=Path(row['paths']['native_metadata']).parent.parent/'conditions/preflight/root_motion.npz'
 with np.load(row['paths']['runtime']) as z:duration=float(z['timestamps'][-1]-z['timestamps'][0])
 with np.load(root) as z:lo=float(z['frame_times'][0]);hi=min(duration,float(z['frame_times'][-1]))
 kept=[f for f in clip['frames'] if max(0,lo)<=f['time_seconds']<=hi]
 # Current source-clock renderer starts at the first real exposure. Never rebase it.
 assert len(kept)>=2 and kept==clip['frames'][:len(kept)] and kept[0]['time_seconds']==0
 if len(kept)==len(clip['frames']):print('All real exposures are supported');return
 prior=out/'before_time_support';prior.mkdir(exist_ok=False)
 for name in ['clip.json','mask_candidates.json']:
  if (out/name).exists():(prior/name).write_bytes((out/name).read_bytes())
 if (out/'ours').exists():(out/'ours').rename(prior/'ours')
 scene_support=read(clip['temporal_support']) if clip.get('temporal_support') else None
 if scene_support:assert sha(clip['temporal_support'])==clip['temporal_support_sha256']
 source_count=scene_support['source_real_frame_count'] if scene_support else len(clip['frames'])
 record=dict(id=sid,status='shared_native_time_interval_selected_before_scores',
  full_capture=str(prior/'clip.json'),full_capture_sha256=sha(prior/'clip.json'),
  source_real_frame_count=source_count,common_real_frame_count=len(kept),
  source_pair_count=source_count-1,common_pair_count=len(kept)-1,
  common_supported_interval_seconds=[max(0,lo),hi],
  excluded_real_exposures=[dict(index=f['index'],timestamp_us=f['timestamp_us'],time_seconds=f['time_seconds'],
   missing_method='ours',reason='outside_native_runtime_or_root_time_support') for f in clip['frames'][len(kept):]]+(scene_support['excluded_real_exposures'] if scene_support else []),
  runtime=str(row['paths']['runtime']),runtime_sha256=sha(row['paths']['runtime']),
  root_motion=str(root),root_motion_sha256=sha(root),
  rule='All source exposures in common native time support; no quality, mask area or score selection',
  scene_native_time_support='OmniRe and SplatAD visibility/time brackets verified for kept exposures by prepare',
  scene_temporal_support=clip.get('temporal_support'),
  time_rebased=False,loop_hold_or_interpolated_reference_used=False,full_capture_preserved=True,
  script_sha256=sha(__file__))
 save(out/'temporal_support.json',record)
 clip.update(frames=kept,real_frame_count=len(kept),source_real_frame_count=record['source_real_frame_count'],
  duration_seconds=kept[-1]['time_seconds'],temporal_support=str(out/'temporal_support.json'),
  temporal_support_sha256=sha(out/'temporal_support.json'),status='common_native_time_interval_ready')
 save(out/'clip.json',clip)
 if (out/'mask_candidates.json').exists():
  candidates=read(out/'mask_candidates.json');candidates['frames']=candidates['frames'][:len(kept)]
  candidates.update(clip_sha256=sha(out/'clip.json'),full_capture_candidates=str(prior/'mask_candidates.json'))
  save(out/'mask_candidates.json',candidates)
 print(json.dumps(dict(common_frames=len(kept),source_frames=record['source_real_frame_count'],excluded=record['excluded_real_exposures'])))


def ours(sid,out):
 import torch
 from export_ours_frozen_time_v1 import LibraryAsset,load_asset
 from render_ours_source_pilot_v1 import render
 active,row=selected(sid);clip=read(out/'clip.json');paths=row['paths'];meta=read(paths['asset_metadata'])
 runtime=load_asset(LibraryAsset(sid,'human','human',Path(paths['runtime']),meta,(Path(paths['asset_metadata']),)))
 cfg=read(paths['production_config']);man=read(cfg['provenance']['manifest']);root=Path(paths['native_metadata']).parent.parent/'conditions/preflight/root_motion.npz'
 with np.load(root) as z:times=z['frame_times'];transforms=z['local_to_smpl_incam']
 normal=np.array(man['source_to_normalized']);view0=np.array(next(v for v in man['cameras'] if v['view']==0)['camtoworld']);rot=Slerp(times,Rotation.from_matrix(transforms[:,:3,:3]));a=runtime.arrays
 folder=out/'ours';folder.mkdir(exist_ok=False);rows=[]
 with torch.inference_mode():
  for f in clip['frames']:
   t=f['time_seconds'];cloud=runtime.sample(t);query=float(a['timestamps'][0])+t
   hi=min(int(np.searchsorted(a['timestamps'],query,side='right')),len(a['timestamps'])-1);lo=max(0,hi-1);fraction=(query-a['timestamps'][lo])/(a['timestamps'][hi]-a['timestamps'][lo])
   def interp(k):return (1-fraction)*a[k][lo]+fraction*a[k][hi]
   xy=interp('root_xy');yaw=float(interp('root_yaw'));c,s=np.cos(-yaw),np.sin(-yaw);inplace=np.eye(4);inplace[:2,:2]=[[c,-s],[s,c]];inplace[:2,3]=-inplace[:2,:2]@xy;inplace[2,3]=float(a['ground_offset'])
   original=cloud.transformed(np.linalg.inv(inplace@a['asset_to_standard']))
   hi=min(int(np.searchsorted(times,t,side='right')),len(times)-1);lo=max(0,hi-1);fraction=(t-times[lo])/(times[hi]-times[lo])
   T=np.eye(4);T[:3,:3]=rot([t]).as_matrix()[0];T[:3,3]=(1-fraction)*transforms[lo,:3,3]+fraction*transforms[hi,:3,3]
   T=T@np.linalg.inv(normal)@view0
   rgb,alpha=render(original.transformed(T),np.eye(4),np.array(f['K_raw']),f['render_wh'])
   paths={}
   for label,ar in [('rgb',rgb),('alpha',alpha)]:
    p=folder/f"{f['index']:05d}_{label}.png";Image.fromarray(np.floor(ar*255+.5).astype(np.uint8)).save(p);paths[label]=str(p);paths[label+'_sha256']=sha(p)
   rows.append(dict(index=f['index'],timestamp_us=f['timestamp_us'],**paths));print('ours',f['index'],flush=True)
 save(folder/'validation.json',dict(status='source_dynamic_frames_rendered_pending_joint_comparison',method='ours',id=sid,frames=rows,
  non_looping=True,speed_ratio=1,full_sh_degree=3,prediction_encoding='white_composited_uint8',source_pose_estimated=True,
  clip_sha256=sha(out/'clip.json'),runtime_sha256=sha(row['paths']['runtime']),script_sha256=sha(__file__),reference_metric_eligible=False))

def review(sid,out):
 import subprocess,re
 import imageio_ffmpeg
 from human_source_metrics_v1 import reference_roi,crop
 archive=out/'source_archive';archive.mkdir(exist_ok=True)
 (archive/(sha(__file__)+'.py')).write_bytes(Path(__file__).read_bytes())
 reviewed=(out/'reference_qualification.json').exists()
 clip=read(out/'clip.json');masks=read(out/('reference_qualification.json' if reviewed else 'mask_candidates.json'));methods=['ours','omnire','splatad'];results={m:read(out/m/'validation.json') for m in methods}
 if reviewed:
  assert masks['reference_eligible'] and masks['clip_sha256']==sha(out/'clip.json')
  masks=dict(masks,frames=[dict(r,path=r['mask'],sha256=r['mask_sha256']) for r in masks['frames']])
 union=np.zeros((900,1600),np.uint8)
 for r in masks['frames']:
  assert sha(r['path'])==r['sha256'];union|=np.array(Image.open(r['path']))
 roi=reference_roi(union);folder=out/('common_roi_reviewed' if reviewed else 'common_roi_review');folder.mkdir(exist_ok=True)
 for f,maskrec in zip(clip['frames'],masks['frames']):
  assert f['index']==maskrec['index'] and f['timestamp_us']==maskrec['timestamp_us']
  rgb=np.array(Image.open(f['source_rgb']).convert('RGB'));mask=np.array(Image.open(maskrec['path']))>127
  reference=crop(np.where(mask[...,None],rgb,255).astype(np.uint8),roi);panels=[('Reference (reviewed)' if reviewed else 'Reference mask candidate',reference)]
  for method in methods:
   r=results[method]['frames'][f['index']];assert r['timestamp_us']==f['timestamp_us'] and sha(r['rgb'])==r['rgb_sha256']
   panels.append((method,crop(np.array(Image.open(r['rgb']).convert('RGB')),roi)))
  tile=Image.new('RGB',(512*len(panels),544),'white');draw=ImageDraw.Draw(tile)
  for j,(label,array) in enumerate(panels):tile.paste(Image.fromarray(array),(512*j,32));draw.text((512*j+8,8),f"{label} | t={f['time_seconds']:.3f}s",fill='black')
  tile.save(folder/f"{f['index']:05d}.png")
 concat=folder/'frames.ffconcat';lines=['ffconcat version 1.0']
 for i,f in enumerate(clip['frames']):
  duration=(clip['frames'][i+1]['timestamp_us']-f['timestamp_us'])/1e6 if i+1<len(clip['frames']) else .001
  lines.extend([f"file '{folder/f'{i:05d}.png'}'",'option framerate 1000000',f'duration {duration:.6f}'])
 concat.write_text('\n'.join(lines)+'\n');ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();video=out/('comparison.mp4' if reviewed else 'internal_dynamic_comparison.mp4')
 subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-safe','0','-i',str(concat),'-fps_mode','vfr','-enc_time_base','1/1000000','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-video_track_timescale','1000000',str(video)],check=True)
 probe=subprocess.run([ffmpeg,'-hide_banner','-i',str(video),'-vf','showinfo','-f','null','-'],capture_output=True,text=True,check=True)
 # Use integer PTS: showinfo's decimal pts_time text rounds at larger times.
 timebase=re.search(r'config in time_base:\s*(\d+)/(\d+)',probe.stderr);assert timebase
 pts=[int(x)*int(timebase[1])/int(timebase[2]) for x in re.findall(r'\bn:\s*\d+\s+pts:\s*(\d+)',probe.stderr)]
 np.testing.assert_allclose(pts,[f['time_seconds'] for f in clip['frames']],atol=1e-6,rtol=0)
 contact=Image.new('RGB',(1536,408*3),'white')
 for j,i in enumerate([0,len(clip['frames'])//2,len(clip['frames'])-1]):contact.paste(Image.open(folder/f'{i:05d}.png').resize((1536,408)),(0,j*408))
 contact_path=out/('comparison_start_middle_end.jpg' if reviewed else 'internal_start_middle_end.jpg');contact.save(contact_path)
 save(out/('comparison.json' if reviewed else 'review_manifest.json'),dict(status='reviewed_real_exposure_dynamic_video_created' if reviewed else 'internal_diagnostic_dynamic_video_created',frames=len(pts),frame_timestamp_error_seconds_max=float(np.max(np.abs(np.array(pts)-[f['time_seconds'] for f in clip['frames']]))),
  fixed_roi_xyxy=roi,roi_basis='union of reviewed exact-exposure reference masks; no prediction bbox' if reviewed else 'union of exact-exposure reference mask candidates; no prediction bbox',methods=methods,
  video=str(video),video_sha256=sha(video),real_exposure_times_preserved=True,last_frame_hold_seconds=.001,
  reference_review='Reviewed actual source masks; full person and carried item retained' if reviewed else 'Mask candidates pending review',
  reference_qualification_sha256=sha(out/'reference_qualification.json') if reviewed else None,
  temporal_metrics_produced=(out/'metrics/metrics.json').exists(),final_gallery_accepted=False,script_sha256=sha(__file__)))
 if reviewed and (out/'metrics/metrics.json').exists():
  import base64,html
  scores=read(out/'metrics/metrics.json');assert list(roi)==scores['fixed_roi_xyxy'] and not scores['failures']
  names=['PSNR','SSIM','LPIPS','IoU','tOF','tLP100'];cells=[]
  for method in methods:
   values=scores['methods'][method]['summary']
   cells.append('<tr><th>'+html.escape(method)+'</th>'+''.join('<td>'+format(values[n]['value'],'.4f')+'</td>' for n in names)+'</tr>')
  doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HN0004 人体动态对比</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:17px/1.65 sans-serif;color:#20252b;background:#fafafa}video,img{width:100%;background:white}table{border-collapse:collapse;background:white;width:100%}th,td{padding:10px;border:1px solid #ddd;text-align:right}th:first-child{text-align:left}p{max-width:1100px}small{color:#555}</style><h1>HN0004 · 同一真实时间段的动态人体对比</h1><p>左到右：真实参考、Ours、OmniRe、SplatAD。30个真实曝光时刻，原始时长2.4秒；同一时刻使用同一源相机、整段固定裁剪区域与白底。视频保持真实时间间隔，不循环、不补帧。SplatAD保留原生整体运动；Ours和OmniRe保留各自的非刚体变化。</p>'''
  doc+='<video controls playsinline preload="metadata" src="data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()+'"></video>'
  doc+='<p>这是1人的源观测重建样例，尚不能作为100人总体排名。全部方法使用30帧及29个相邻帧对。参考mask包含背包与可见手持物，模糊边缘的不确定性保留。</p><table><tr><th>方法</th>'+''.join('<th>'+n+(' ↑' if n in ['PSNR','SSIM','IoU'] else ' ↓')+'</th>' for n in names)+'</tr>'+''.join(cells)+'</table>'
  doc+='<p>tOF比较真实与预测的光流；tLP100比较两者相邻帧LPIPS变化。二者须结合逐帧质量解读，平滑不代表动作正确。采用本实验固定ROI和AlexNet v0.1协议，不能直接与其他论文数值横比。<a href="https://github.com/thunil/TecoGAN/blob/master/metrics.py">公式参考：TecoGAN官方实现</a>。</p><p>首帧、中间帧与尾帧：</p>'
  doc+='<img alt="真实参考与三个方法的首、中、尾时刻" src="data:image/jpeg;base64,'+base64.b64encode(contact_path.read_bytes()).decode()+'"><p><a href="metrics/summary.csv">逐方法指标CSV</a> · <a href="metrics/metrics.json">逐帧、逐帧对和对照实验JSON</a></p></html>'
  doc=doc.replace('HN0004',sid)
  doc=doc.replace('30个真实曝光时刻，原始时长2.4秒',f"{len(clip['frames'])}个真实曝光时刻，原始时长{clip['duration_seconds']:g}秒")
  doc=doc.replace('30帧及29个相邻帧对',f"{len(clip['frames'])}帧及{len(clip['frames'])-1}个相邻帧对")
  doc=doc.replace('参考mask包含背包与可见手持物','参考mask保留衣物、穿戴物与可见携带物')
  if clip.get('temporal_support'):
   temporal=read(clip['temporal_support']);assert sha(clip['temporal_support'])==clip['temporal_support_sha256']
   note=f"<p>时间覆盖：共同支持{temporal['common_real_frame_count']}/{temporal['source_real_frame_count']}个真实曝光、{temporal['common_pair_count']}/{temporal['source_pair_count']}个原始相邻帧对。完整采集保留；超出原生时间范围的曝光另记缺测，不循环、不复制末帧，不因质量或遮挡删帧。<a href='temporal_support.json'>时间覆盖记录</a>。</p>"
   doc=doc.replace('<video controls',note+'<video controls',1)
  (out/'index.html').write_text(doc)
 print(json.dumps(dict(video=str(video),frames=len(pts),roi=roi)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--human-id',default='HN0004');p.add_argument('action',choices=['prepare','support','masks','ours','review']);a=p.parse_args();out=H/'reports/dynamic_source_clip_v1'/a.human_id
 globals()[a.action](a.human_id,out)
