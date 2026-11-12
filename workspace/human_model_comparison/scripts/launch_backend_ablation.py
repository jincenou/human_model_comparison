"""Run unchanged native/optimized backend stages on verified original caches."""
import argparse
import fcntl
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from restore_frontend_rgba_cache import H,CODE,read,save,sha


def check_inputs(root,sid):
    protocol=read(root/'protocol.json')
    for path,digest in protocol['code_hashes'].items():assert sha(path)==digest,path
    assert sha(protocol['restored_manifest'])==protocol['restored_manifest_sha256']
    manifest=read(protocol['restored_manifest'])
    for row in manifest['input_hashes']:
        path=Path(manifest['source_data'])/'images'/f'cam{row["view"]:02d}'/f'{row["frame"]:06d}.png'
        assert sha(path)==row['sha256'],path
    assert sha(root/'data'/sid/'canonical.npz')==protocol['initialization_sha256']
    return protocol


def worker(root,sid,stage):
    # Use the production CLI, including its heldout-file audit hook. The only
    # execution adjustment is the user-requested 10 GiB resource floor.
    protocol=check_inputs(root,sid);sys.path.insert(0,str(CODE))
    os.environ['HUMAN_RUN_ROOT']=str(root)
    import common
    common.disk_check.__defaults__=(10,)
    import torch
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    sys.argv=[str(CODE/'reconstruction.py'),stage,sid]
    runpy.run_path(str(CODE/'reconstruction.py'),run_name='__main__')
    stage_dir=root/'outputs'/sid/({'native':'native','alpha':'alpha','A':'asset'}[stage])
    complete=read(stage_dir/'complete.json');expected={'native':32999,'alpha':47999,'A':52999}[stage]
    assert complete['actual_updates']==expected
    save(stage_dir/'ablation_execution.json',dict(stage=stage,actual_updates=expected,wall_seconds_including_loading=time.monotonic()-start,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),torch_version=torch.__version__,cuda_version=torch.version.cuda,protocol_sha256=sha(root/'protocol.json'),production_code_hashes=protocol['code_hashes'],numerical_training_code_modified=False,disk_floor_gib=10))


def main():
    p=argparse.ArgumentParser();p.add_argument('--human-id',required=True);p.add_argument('--gpu',type=int,default=2);p.add_argument('--stages',nargs='+',choices=['native','alpha','A'],default=['native']);p.add_argument('--worker',action='store_true');a=p.parse_args()
    root=H/'runs/backend_ablation_v1'/a.human_id;assert root.exists()
    if a.worker:
        assert len(a.stages)==1;worker(root,a.human_id,a.stages[0]);return
    policy=read(H/'state/gpu_resource_policy.json');assert a.gpu in policy['allowed_gpu_indices']and a.gpu not in policy['excluded_gpu_indices']
    statepath=root/('controller_'+'_'.join(a.stages)+'.json');assert not statepath.exists(),'Inspect previous controller before restarting'
    check_inputs(root,a.human_id)
    with (H/'state'/f'gpu_auxiliary_slot_{a.gpu}.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        info=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',');uuid=info[0].strip();assert int(info[1])>=32768
        assert shutil.disk_usage(H).free>=10*1024**3
        sys.path.insert(0,str(CODE));os.environ['HUMAN_RUN_ROOT']=str(root);import common
        env=common.environment(a.gpu,'grid4d');env['HUMAN_RUN_ROOT']=str(root)
        state=dict(status='running',id=a.human_id,pid=os.getpid(),gpu=a.gpu,gpu_uuid=uuid,stages=[],started_at=time.time(),protocol_sha256=sha(root/'protocol.json'),script_sha256=sha(__file__))
        save(statepath,state)
        for stage in a.stages:
            command=[str(common.PYTHONS['grid4d']),'-B','-u',str(Path(__file__).resolve()),'--worker','--human-id',a.human_id,'--stages',stage]
            entry=dict(stage=stage,status='running',command=command,started_at=time.time());state['stages'].append(entry)
            log=root/(stage+'.log')
            with log.open('x')as stream:child=subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lease.fileno(),))
            entry['pid']=child.pid;save(statepath,state);code=child.wait();entry.update(exit_code=code,status='finished'if code==0 else'failed_retained',finished_at=time.time());save(statepath,state)
            if code:
                state.update(status='failed_retained',finished_at=time.time());save(statepath,state);raise RuntimeError(stage+' failed; inspect retained log and checkpoint')
        state.update(status='stages_finished_pending_evaluation',finished_at=time.time());save(statepath,state)


if __name__=='__main__':main()
