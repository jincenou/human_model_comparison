"""Matched 20k continuation: exact production CLI with alpha weight set to zero.

The original 33k checkpoint and cached frontend are shared read-only. All new
training outputs live in a distinct run root; production code is never edited.
"""
import argparse
import difflib
import fcntl
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common_human_orbit import H, read, save, sha, checked
from launch_backend_ablation import CODE, check_inputs

OLD = '            loss = rgb_loss + 0.1 * a_loss + reg'
NEW = '            loss = rgb_loss + 0.0 * a_loss + reg'


def roots(sid):
    parent = H/'runs/backend_ablation_v1'/sid
    return parent, parent/'rgb_continuation_53k'


def prepare(sid):
    parent, root = roots(sid)
    check_inputs(parent, sid)
    source = (CODE/'reconstruction.py').read_text()
    assert source.count(OLD) == 1 and NEW not in source
    edited = source.replace(OLD, NEW)
    compile(edited, str(root/'reconstruction_rgb_control.py'), 'exec')
    assert read(parent/'outputs'/sid/'native/complete.json')['actual_updates'] == 32999
    for stage in ['alpha', 'asset']:
        assert (parent/'outputs'/sid/stage/'complete.json').exists()
    root.mkdir(exist_ok=False)
    (root/'config').symlink_to(parent/'config', target_is_directory=True)
    (root/'data').symlink_to(parent/'data', target_is_directory=True)
    native = root/'outputs'/sid/'native'; native.parent.mkdir(parents=True)
    native.symlink_to(parent/'outputs'/sid/'native', target_is_directory=True)
    local = root/'reconstruction_rgb_control.py'; local.write_text(edited)
    (root/'coefficient_change.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), edited.splitlines(True),
        fromfile='production/reconstruction.py', tofile='control/reconstruction_rgb_control.py')))
    save(root/'protocol.json',dict(id=sid,variant='rgb_continuation_53k',
        parent_protocol=str(parent/'protocol.json'),parent_protocol_sha256=sha(parent/'protocol.json'),
        source=str(CODE/'reconstruction.py'),source_sha256=sha(CODE/'reconstruction.py'),
        local_source=str(local),local_source_sha256=sha(local),
        native_recovery=str(native/'recovery.pt'),native_recovery_sha256=sha(native/'recovery.pt'),
        initial_actual_updates=32999,continuation_updates=[15000,5000],final_actual_updates=52999,
        alpha_loss_weight=0.0,optimized_comparator_alpha_weight=0.1,
        change='Exactly one coefficient changed; RGB loss, regularizers, rendering, schedules, sampling and initialization retained',
        input_cache_regenerated=False,production_assets_modified=False,
        comparison_scope='Alpha-loss contribution at matched staged budget; not the unmodified original33k recipe'))
    print(root, flush=True)


def verify(sid):
    parent, root = roots(sid); p = read(root/'protocol.json')
    checked(p['parent_protocol'],p['parent_protocol_sha256'])
    checked(p['source'],p['source_sha256'])
    checked(p['local_source'],p['local_source_sha256'])
    checked(p['native_recovery'],p['native_recovery_sha256'])
    assert Path(p['local_source']).read_text() == Path(p['source']).read_text().replace(OLD, NEW)
    check_inputs(parent,sid)
    return p


def worker(sid, stage):
    parent, root = roots(sid); protocol = verify(sid)
    sys.path.insert(0,str(CODE));os.environ['HUMAN_RUN_ROOT']=str(root)
    import common
    common.disk_check.__defaults__=(10,)
    import torch
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    sys.argv=[protocol['local_source'],stage,sid]
    runpy.run_path(protocol['local_source'],run_name='__main__')
    dirname='alpha' if stage=='alpha' else 'asset'
    run=root/'outputs'/sid/dirname; comparison=parent/'outputs'/sid/dirname
    actual=47999 if stage=='alpha' else 52999
    assert read(run/'complete.json')['actual_updates']==actual
    assert read(run/'complete.json')['points']==read(comparison/'complete.json')['points']
    equal_files={}
    for name in ['sampling.json','learning_rates.json']:
        assert sha(run/name)==sha(comparison/name),name
        equal_files[name]=sha(run/name)
    # Check that the shared starting checkpoint was not changed by either stage.
    checked(protocol['native_recovery'],protocol['native_recovery_sha256'])
    save(run/'ablation_execution.json',dict(stage=stage,actual_updates=actual,
        wall_seconds_including_loading=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],torch_version=torch.__version__,
        alpha_loss_weight=0.0,matched_comparator_files=equal_files,
        protocol_sha256=sha(root/'protocol.json'),script_sha256=sha(__file__),
        heldout_access_guard_preserved=True,disk_floor_gib=10))


def main():
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--worker',choices=['alpha','A'])
    p.add_argument('--gpu',type=int,default=2);a=p.parse_args()
    if a.prepare:prepare(a.human_id);return
    if a.worker:worker(a.human_id,a.worker);return
    parent,root=roots(a.human_id);verify(a.human_id)
    policy=read(H/'state/gpu_resource_policy.json')
    assert a.gpu in policy['allowed_gpu_indices'] and a.gpu not in policy['excluded_gpu_indices']
    path=root/'controller.json';assert not path.exists(),'Inspect previous execution; no automatic retry'
    with (H/'state'/f'gpu_auxiliary_slot_{a.gpu}.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert int(info[1])>=32768 and shutil.disk_usage(H).free>=15*1024**3
        sys.path.insert(0,str(CODE));os.environ['HUMAN_RUN_ROOT']=str(root);import common
        env=common.environment(a.gpu,'grid4d');env['HUMAN_RUN_ROOT']=str(root)
        state=dict(status='running',id=a.human_id,pid=os.getpid(),gpu=a.gpu,gpu_uuid=info[0].strip(),
            stages=[],started_at=time.time(),protocol_sha256=sha(root/'protocol.json'),script_sha256=sha(__file__))
        save(path,state)
        for stage in ['alpha','A']:
            command=[str(common.PYTHONS['grid4d']),'-B','-u',str(Path(__file__).resolve()),'--human-id',a.human_id,'--worker',stage]
            record=dict(stage=stage,status='running',command=command,started_at=time.time());state['stages'].append(record)
            with (root/(stage+'.log')).open('x') as stream:
                child=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
            record['pid']=child.pid;save(path,state);code=child.wait()
            record.update(exit_code=code,status='finished' if code==0 else 'failed_retained',finished_at=time.time());save(path,state)
            if code:
                state.update(status='failed_retained',finished_at=time.time());save(path,state)
                raise RuntimeError('Control stage failed; inspect retained outputs before any restart')
        state.update(status='stages_finished_pending_evaluation',finished_at=time.time());save(path,state)


if __name__=='__main__':main()
