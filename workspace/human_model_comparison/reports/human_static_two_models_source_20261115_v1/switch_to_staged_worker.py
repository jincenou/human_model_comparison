from human_asset_benchmark.paths import expand as _hmc_expand
import sys,os,time,json,signal,subprocess,psutil
from pathlib import Path
H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'));sys.path.insert(0,str(H/'scripts'))
import evaluate_human_static_two_models as src
from run_human_static_two_models_staged import gpu_lease
O=src.OUT;oldpid=3533267;gpu=1
src.save(O/'controller_transition.json',dict(status='waiting_for_old_group_to_finish',old_pid=oldpid,gpu=gpu,pid=os.getpid(),started_at=time.time()))
with gpu_lease(gpu):
    process=psutil.Process(oldpid) if psutil.pid_exists(oldpid) else None
    for _ in range(100):
        running=[j for j in src.read(O/'queue.json')['jobs'] if j['status']=='running' and j.get('pid')==oldpid]
        children=process.children(recursive=True) if process and process.is_running() else []
        if not running and not children:break
        time.sleep(.2)
    assert not running and not children,('Cannot safely switch active work',running,[x.pid for x in children])
    if process and process.is_running():
        cmd=process.cmdline();assert str(src.S) in cmd and 'worker' in cmd
        process.terminate();process.wait(timeout=10)
    cmd=[src.PY['control'],'-B','-u',str(H/'scripts/run_human_static_two_models_staged.py'),'--gpu','1','--batch-size','4']
    with (O/'batch_gpu1_staged.log').open('a') as f:p=subprocess.Popen(cmd,cwd=H,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    src.save(O/'batch_gpu1_staged_launcher.json',dict(pid=p.pid,command=cmd,started_at=time.time()))
    src.save(O/'controller_transition.json',dict(status='completed',old_pid=oldpid,new_pid=p.pid,gpu=gpu,finished_at=time.time(),proof='Acquired all three GPU locks after old group release; no running old jobs or descendants before terminating idle controller',terminated_generation_or_render_process=False))
    print('staged_pid',p.pid,flush=True)
