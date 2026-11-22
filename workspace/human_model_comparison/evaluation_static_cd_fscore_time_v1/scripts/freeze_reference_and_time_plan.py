"""Reference-only review and metadata-only timing selection before model scores."""
import hashlib
from collections import Counter
from common import *

def key(s):return hashlib.sha256(('human-time-v1:'+s).encode()).hexdigest()

def main():
    active,selected=selection();candidate=read(E/'references/reference_candidates.json')
    special={
      'HN0215':'Umbrella overhead; pedestrian label/annotation extent may include carried umbrella. Scope is labelled target returns and whole annotation extent, not anatomical body height.',
      'HN0190':'Umbrella overhead and wet scene; retain nominal scan timing and whole labelled target extent. No pose-equality claim.',
      'HN0397':'Sitting/lying attribute; annotation extent measures seated target and carried bag, not standing height.',
      'HN0144':'Foreground fence in source image; semantically filtered LiDAR target points remain attributable. Image occlusion is not geometry rejection.',
      'HN0145':'Source foreground railing crosses legs; points remain target-attributable under label and box check.',
      'HN0310':'Visible backpack and walking limb displacement; projected scan returns agree with target, with near-boundary temporal offsets.',
      'HN0004':'Visible red backpack; class labels are not an exhaustive accessory segmentation.',
      'HN0646':'Replacement identity and neighbour checked; sparse 10-query support remains exploratory.',
      'HN0266':'Multiple neighbouring people; retained returns project to centre target, same-class overlapping-box points were removed.',
      'HN0252':'Two neighbouring people; retained points project to intended left target, no identity swap observed.',
      'HN0217':'Two neighbouring people; scan points align with intended right target.',
      'HN0172':'Two neighbouring people; scan points align with white-shirt target.'}
    reviews=[];refs=[]
    for i,r in enumerate(candidate['rows']):
        p=E/'references/review'/f'sheet_{i//20+1:02d}.jpg'
        decision=dict(id=r['id'],reviewer='Codex direct visual inspection in this conversation',independent_human_blind_review=False,
          evidence=str(p),evidence_sha256=sha(p),tile_index=i%20,decision='accepted_observed_target_nominal_scan_proxy',
          identity='No gross wrong-person/background attribution visible in raw-camera projection; exact instance token and sensor transform independently checked.',
          time='No gross pose/identity inconsistency visible. 0.2–44.879 ms timestamp mismatch and within-scan motion remain unbounded errors; no guarantee of sub-10cm correspondence.',
          scope=special.get(r['id'],'Projected returns follow intended dressed target. Sparse unobserved surfaces and carried-object label coverage are not certified.'),
          model_scores_consulted=False)
        reviews.append(decision);refs.append(dict(r,reference_status='qualified_nominal_observed_proxy',reference_qualified=True,review=decision))
    immutable(E/'references/visual_review.json',dict(candidate_sha256=sha(E/'references/reference_candidates.json'),rows=reviews))
    immutable(E/'reference_manifest.json',dict(selection_sha256=SELECTION_SHA,rows=refs,rules=candidate['rules'],
        status='reference_frozen_before_geometry_scores',reference_source_sha256=sha(E/'references/reference_candidates.json'),review_sha256=sha(E/'references/visual_review.json')))
    population=[]
    for sid,s in selected.items():
        r=next(x for x in refs if x['id']==sid)
        b=s['reference_frame']['bbox_input_xyxy']
        population.append(dict(id=sid,duration_s=s['input_budget']['native_duration_s'],real_exposures=s['input_budget']['real_native_exposures'],
          native_pixel_height=b[3]-b[1],pose=';'.join(r['attribute_names']) or 'unknown',hash=key(sid)))
    # Split at the median duration/exposure rank, then median pixel height.
    # Ties use ID hash. Within each stratum choose a pose category by rotating
    # sorted available metadata categories, then the minimum ID hash. This
    # uses no asset success, LiDAR support, old score, or geometry score.
    ordered=sorted(population,key=lambda r:(r['duration_s'],r['real_exposures'],r['hash']))
    members=[]
    for dh in range(2):
        half=sorted(ordered[dh*50:(dh+1)*50],key=lambda r:(r['native_pixel_height'],r['hash']))
        for ph in range(2):
            group=half[ph*25:(ph+1)*25];poses=sorted({r['pose'] for r in group});index=2*dh+ph;pose=poses[index%len(poses)]
            member=min((r for r in group if r['pose']==pose),key=lambda r:r['hash'])
            members.append(dict(member,stratum=index,duration_half=dh,pixel_height_half=ph,available_poses=poses,selected_pose=pose))
    runs=[]
    for m in METHODS[:4]:
        for item in members:runs.append(dict(method=m,id=item['id'],repeat=0,status='queued'))
        runs.append(dict(method=m,id=members[0]['id'],repeat=1,status='queued'))
    immutable(E/'time_plan.json',dict(selection_sha256=SELECTION_SHA,status='frozen_before_timing',population_metadata=population,objects=members,runs=runs,
      selection_rule='Hierarchical median ranks of duration/exposure count then native raw pixel height; pose-category rotation within each of four strata; salted SHA256 ID minimum; ties by ID hash.',
      total_formal_attempts=20,anchor=members[0]['id'],seed=42,ours_updates=[32999,15000,5000],
      no_replacement_on_failure=True,scene_methods='not applicable / not measured this round',timing_condition='stage-preloaded complete required generation; include preprocessing, I/O and native saving; exclude model initialization and independent reload',
      fairness='same GPU model and quality configuration; exclusive task leases; log and review CPU/storage/GPU contention'))
    print('References frozen',dict(Counter(r['candidate_panel'] for r in refs)),'Time IDs',[r['id'] for r in members])

if __name__=='__main__':main()
