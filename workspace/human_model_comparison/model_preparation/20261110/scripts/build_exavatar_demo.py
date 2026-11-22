"""Build a local, bounded native-camera demo from verified ExAvatar exports."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
from PIL import Image, ImageDraw

P = Path(__file__).resolve().parents[1]
ROOT = P/'reports/exavatar_real_sequence_HN0057_20261113'
ASSETS = ROOT/'native_asset_reload'
OUT = P.parents[1]/'reports/new_model_demos_20261113/exavatar_HN0057'
OUT.mkdir(parents=True, exist_ok=True)
frames_out = OUT/'motion_frames'
frames_out.mkdir(exist_ok=True)
manifest = json.loads((ASSETS/'export.json').read_text())
reload = json.loads((ASSETS/'reload.json').read_text())
assert len(manifest['frames']) == 30 and len(reload['frames']) == 60
rows = sorted(manifest['frames'], key=lambda r:r['frame'])
panels = []
times = [r['timestamp_us'] for r in rows]
durations = [(b-a)/1000 for a,b in zip(times,times[1:])]
durations.append(durations[-1])
for r in rows:
    i = r['frame']
    inp = Image.open(ROOT/'frames'/f'{i}.png').convert('RGB')
    pred = Image.open(ASSETS/f'{i:03d}_refined_reloaded.png').convert('RGB')
    assert inp.size == pred.size
    panel = Image.new('RGB',(inp.width*2,inp.height+24),'white')
    draw = ImageDraw.Draw(panel)
    draw.text((8,5),'Input',fill='black');draw.text((inp.width+8,5),'ExAvatar',fill='black')
    panel.paste(inp,(0,24));panel.paste(pred,(inp.width,24))
    panel.save(frames_out/f'{i:03d}.png')
    panels.append(panel)
panels[17].save(OUT/'source_comparison.png')
panels[0].save(OUT/'motion.gif',save_all=True,append_images=panels[1:],
               duration=[round(d/10)*10 for d in durations],loop=0,disposal=2)
orbit = [Image.open(ASSETS/'orbit'/f'{i:02d}.png').convert('RGB')for i in range(16)]
orbit[0].save(OUT/'orbit.gif',save_all=True,append_images=orbit[1:],duration=150,loop=0,disposal=2)
shutil.copy2(ASSETS/'orbit_review.jpg',OUT/'side_back.jpg')
concat = []
for r,duration in zip(rows,durations):
    concat += [f"file 'motion_frames/{r['frame']:03d}.png'", 'option framerate 1000000', f'duration {duration/1000:.6f}']
concat.append(f"file 'motion_frames/{rows[-1]['frame']:03d}.png'")
concat.append('option framerate 1000000')
(OUT/'motion_concat.txt').write_text('\n'.join(concat)+'\n')
video_status = 'not_encoded'
try:
    import imageio_ffmpeg
    command = [imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','concat','-safe','0',
               '-i','motion_concat.txt','-fps_mode','vfr','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',
               '-movflags','+faststart','motion.mp4']
    result = subprocess.run(command,cwd=OUT,capture_output=True,text=True,timeout=60)
    (OUT/'video_encode.log').write_text(result.stderr)
    video_status = 'encoded_vfr' if result.returncode == 0 else 'encode_failed_gif_available'
except (ImportError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as error:
    (OUT/'video_encode.log').write_text(repr(error))
motion = '<video controls loop muted playsinline poster="source_comparison.png" src="motion.mp4"></video>' if video_status == 'encoded_vfr' else '<img src="motion.gif" alt="真实输入与ExAvatar动作" />'
html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ExAvatar HN0057 demo</title><style>body{font:16px system-ui;margin:28px auto;padding:0 20px;max-width:1150px;color:#20252b;background:#f5f6f8}h1{font-size:25px}p{line-height:1.6;color:#525963}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;padding:18px;border-radius:12px}img,video{max-width:100%;display:block;margin:auto}video{height:440px}.source{height:440px}h2{font-size:18px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>ExAvatar · HN0057</h1><p>30帧真实输入 · 2250次训练更新 · 白底人体独立重载完成</p>
<div class="grid"><section><h2>原动作</h2>''' + motion + '''</section><section><h2>固定姿态环绕</h2><img class="source" src="orbit.gif" alt="ExAvatar固定姿态16方位"></section></div>
<section style="margin-top:20px"><h2>源图与结果</h2><img class="source" src="source_comparison.png" alt="真实输入与ExAvatar"></section>
<section style="margin-top:20px"><h2>正侧背面</h2><img src="side_back.jpg" alt="ExAvatar四方位"></section>
<p>当前为原生相机demo，尚未完成跨模型共同相机标定。源图与人体渲染使用相同画布；环绕单独采用固定中心与距离。侧背面无真实参考，脸手模糊和背面纹理伪影保留。</p>
<p>动作使用原30个真实曝光时刻；预览循环只用于播放，不用于指标。视频按曝光间隔编码，末帧短暂停留；GIF时长舍入至10毫秒。环绕固定第17帧姿态，不是人物转身动作。</p>
<p><a href="motion.gif">动作GIF</a> · <a href="orbit.gif">环绕GIF</a> · <a href="review.json">参数记录</a></p></html>'''
(OUT/'index.html').write_text(html)
review = dict(method='ExAvatar',identity='HN0057',real_frames=30,updates=2250,source_frame=17,
    source_timestamp_us=times[17],timestamps_us=times,video_status=video_status,
    independent_reload_rgb_max_abs=max(r['rgb_max_abs']for r in reload['frames']),
    independent_reload_alpha_max_abs=max(r['alpha_max_abs']for r in reload['frames']),
    common_camera_valid=False,observed_defects=['blurred face and hands','back texture artifacts'],
    status='demo_ready_for_user_review_not_batch_approved',
    file_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in OUT.iterdir()if f.suffix in ['.png','.jpg','.gif','.mp4']})
(OUT/'review.json').write_text(json.dumps(review,indent=2))
print(OUT/'index.html');print(video_status)
