"""Match predictions to the frozen track; preserve native unassigned outputs."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parents[1]
ROOT=P/'reports/exavatar_real_sequence_HN0057_20261113'
rows={r['native_index']:r for r in json.loads((ROOT/'inputs.json').read_text())['frames']}
def iou(a,b):
 a=np.asarray(a).reshape(4);b=np.asarray(b).reshape(4)
 inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
 return float(inter/max(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter,1e-9))
out=ROOT/'keypoints_whole_body_track';out.mkdir(exist_ok=True);audit=[]
for i,row in rows.items():
 f=P/f'sources/exavatar/fitting/tools/mmpose/vis_results/results_{i}.json';d=json.loads(f.read_text())['instance_info'];scores=[iou(x['bbox'],row['bbox_input_xyxy']) for x in d];j=int(np.argmax(scores));assert scores[j]>.3,(i,scores)
 selected=d[j];points=np.column_stack((np.asarray(selected['keypoints']),np.asarray(selected['keypoint_scores'])));assert points.shape==(133,3)
 (out/f'{i}.json').write_text(json.dumps(points.tolist()))
 audit.append(dict(frame=i,selected_instance=j,track_iou=scores[j],candidate_ious=scores,input_predictions=str(f),input_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(ROOT/'dwpose_track_association.json').write_text(json.dumps(audit,indent=2))
source=P/'sources/exavatar/fitting/tools/Hand4Whole_RELEASE/demo/run_hand4whole.py';text=source.read_text();start=text.index('def get_one_box(');end=text.index('\ndef parse_args()',start)
replacement='''def get_one_box(det_output):
    target = np.asarray(track_rows[frame_idx]['bbox_input_xyxy'], dtype=float)
    candidates = []
    for i in range(len(det_output['boxes'])):
        if int(det_output['labels'][i]) != 1 or float(det_output['scores'][i]) < .3:
            continue
        box = det_output['boxes'][i].detach().cpu().numpy()
        inter = np.maximum(np.minimum(box[2:], target[2:])-np.maximum(box[:2], target[:2]), 0).prod()
        union = np.maximum(box[2:]-box[:2], 0).prod()+np.maximum(target[2:]-target[:2], 0).prod()-inter
        candidates.append((float(inter/max(union,1e-9)), box.tolist()))
    assert candidates, ('No person detection', frame_idx)
    overlap, box = max(candidates, key=lambda x:x[0])
    assert overlap > .3, ('No detection matches identity', frame_idx, overlap)
    track_associations.append(dict(frame=frame_idx, track_iou=overlap, selected_detection=box))
    with open(osp.join(root_path, 'hand4whole_track_association.json'), 'w') as f:
        json.dump(track_associations, f, indent=2)
    return box
'''
text=text[:start]+replacement+text[end:]
text=text.replace('root_path = args.root_path','root_path = args.root_path\nwith open(osp.join(root_path, "inputs.json")) as f:\n    track_rows = {r["native_index"]:r for r in json.load(f)["frames"]}\ntrack_associations = []')
text=text.replace("'smplx_init'","'smplx_init_track'").replace("'smplx_init.mp4'","'smplx_init_track.mp4'")
adapter=ROOT/'run_hand4whole_track.py';adapter.write_text(text)
(ROOT/'identity_adapter.json').write_text(json.dumps(dict(upstream=str(source),upstream_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),adapter=str(adapter),adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),policy='Select predicted person detection with largest IoU to existing nuScenes identity box; minimum IoU .3; do not replace predicted boxes, poses, or shapes with annotations.'),indent=2))
print('Prepared identity association for 30 frames')
