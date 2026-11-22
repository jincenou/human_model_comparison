"""Resolve the existing ROI-path alias without changing a camera, image or score rule."""
import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import evaluate_human_static_two_models as src


def main(method, gpu):
    src.freeze();src.configure(method)
    original=src.inputs;used=[]
    def compatible(sid):
        q,cam,payload,result=original(sid)
        if 'reference_rgb' not in cam:
            assert sid=='HN0646' and cam['roi_xyxy']==q['roi_xyxy']
            assert cam['identity_key']==q['identity_key'] and cam['timestamp_us']==q['timestamp_us']
            path=cam['roi_reference_rgb'];digest=cam['roi_reference_rgb_sha256'];src.checked(path,digest)
            historical=src.read(src.R/'reports/source_appearance_alignment_20261115_v1/rows'/f'human__ours__{sid}.json')
            assert historical['reference']['sha256']==digest
            src.checked(historical['reference']['path'],digest)
            cam=dict(cam,reference_rgb=path)
            used.append(dict(id=sid,from_key='roi_reference_rgb',to_key='reference_rgb',path=path,sha256=digest,
                             camera_contract=q['camera_contract'],camera_contract_sha256=q['camera_contract_sha256'],
                             historical_reference_sha256=digest))
        return q,cam,payload,result
    src.inputs=compatible
    with contextlib.ExitStack() as stack:
        if method=='reconviagen':
            for p in [src.P/f'runs/rigid111_gpu_locks/gpu_{gpu}.lock',src.H/f'state/gpu_auxiliary_slot_{gpu}.lock',Path(f'/tmp/drivelab-gpu-locks/gpu{gpu}.lock')]:
                f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX)
        src.views(['HN0646'])
    record=src.ROOT/'observations/HN0646/views.json';data=src.read(record)
    data['reference_metadata_alias_repair']=dict(adapter=str(Path(__file__).resolve()),adapter_sha256=src.sha(__file__),aliases=used,
        scientific_source_unchanged=True,reference_pixels_unchanged=True,transforms_unchanged=True,
        description='Existing camera ROI reference key aliased in memory only; original frozen files untouched.')
    src.save(record,data)
    print(method,'reference metadata alias repaired',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['hunyuan3d21','reconviagen'],required=True);p.add_argument('--gpu',type=int,default=4);a=p.parse_args();main(a.method,a.gpu)
