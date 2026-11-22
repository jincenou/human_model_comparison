"""Stop only the idle parent after its existing CPU child finishes naturally."""
from human_asset_benchmark.paths import expand as _hmc_expand
import sys,time,subprocess,psutil,os
from pathlib import Path
H=Path(_hmc_expand('${HMC_WORKSPACE}/human_model_comparison'));sys.path.insert(0,str(H/'scripts'));import evaluate_human_static_two_models_geometry as geo
src=geo.source;O=geo.OUT;old=psutil.Process(3583978);assert str(geo.S) in old.cmdline() and 'hunyuan3d21' in old.cmdline()
old.suspend()
try:
    while old.status()!=psutil.STATUS_STOPPED:time.sleep(.01)
    descendants=old.children(recursive=True);src.save(O/'cpu_controller_transition.json',dict(status='waiting_existing_CPU_object_to_finish',old_pid=old.pid,children=[p.pid for p in descendants],started_at=time.time(),no_child_suspended=True))
    while True:
        active=[]
        for p in old.children(recursive=True):
            try:
                if p.is_running() and p.status()!=psutil.STATUS_ZOMBIE:active.append(p.pid)
            except psutil.NoSuchProcess:pass
        if not active:break
        time.sleep(1)
    running=[p for p in (O/'results/hunyuan3d21').glob('*.json') if src.read(p)['status']=='running'];assert not running,('CPU child exited without terminal result',running)
    old.terminate();old.resume();old.wait(timeout=10)
except BaseException:
    if old.is_running():old.resume()
    raise
pids=[]
for slot in [0,1]:
    cmd=[src.PY['control'],'-B','-u',str(H/'scripts/run_human_static_geometry_cpu.py'),'--slot',str(slot)]
    with (O/f'hunyuan3d21_cpu{slot}.log').open('a') as f:p=subprocess.Popen(cmd,cwd=H,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    pids.append(p.pid);src.save(O/f'hunyuan3d21_cpu{slot}_launcher.json',dict(pid=p.pid,command=cmd,started_at=time.time()))
src.save(O/'cpu_controller_transition.json',dict(status='completed',old_pid=3583978,new_pids=pids,finished_at=time.time(),proof='Only parent controller paused; existing CPU child finished naturally and published terminal result before controller replacement; per-person flock inherited by new children',native_assets_or_scores_modified=False))
print(pids,flush=True)
