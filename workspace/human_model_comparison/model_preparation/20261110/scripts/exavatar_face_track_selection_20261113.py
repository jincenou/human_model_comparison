"""Associate FAN's own landmarks to the already associated body prediction."""
import json
from pathlib import Path
import numpy as np,cv2,face_alignment
P=Path(__file__).resolve().parents[1];ROOT=P/'reports/exavatar_real_sequence_HN0057_20261113'
model=face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,flip_input=False,device='cuda')
audit=[]
for row in json.loads((ROOT/'inputs.json').read_text())['frames']:
 i=row['native_index'];img=cv2.cvtColor(cv2.imread(str(ROOT/'frames'/f'{i}.png')),cv2.COLOR_BGR2RGB)
 body=np.array(json.loads((ROOT/'keypoints_whole_body_track'/f'{i}.json').read_text()))
 head=body[:5];head=head[head[:,2]>.3];assert len(head)>=2,(i,'insufficient predicted head points')
 anchor=(head[:,:2]*head[:,2:]).sum(0)/head[:,2].sum()
 output=model.get_landmarks_from_image(img,return_bboxes=True);assert output is not None,(i,'no detected face')
 landmarks,boxes=output;scores=[]
 for points,box in zip(landmarks,boxes):
  center=np.asarray(points)[27:36].mean(0);size=max(float(box[2]-box[0]),float(box[3]-box[1]))
  scores.append(float(np.linalg.norm(center-anchor)/size))
 j=int(np.argmin(scores));assert scores[j]<1.0,(i,scores)
 (ROOT/'face_landmarks').mkdir(exist_ok=True)
 np.savetxt(ROOT/'face_landmarks'/f'{i}.txt',landmarks[j])
 audit.append(dict(frame=i,selected_face=j,candidate_distances=scores,anchor_from_predicted_body=anchor.tolist(),landmark_source='Same S3FD/FAN weights as native DECA; identity association only'))
(ROOT/'face_track_association.json').write_text(json.dumps(audit,indent=2));print('faces associated',len(audit),'changed first-face selections',sum(r['selected_face']!=0 for r in audit))
