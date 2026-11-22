"""Resume unexecuted export stages after verified pre-timer wrapper setup failure.

All fifteen actual generation stages stay unchanged. Their measured intervals
are retained, with the failed controller and log archived separately.
"""
import os,subprocess,time,types,shutil
from common import *
CODE=H.parent/'human';PROD=H.parent/'asset_work/human_nuscenes100_20261026'
def main():
    mod=types.ModuleType('native_env');mod.__file__=str(CODE/'common.py');exec(compile((CODE/'common.py').read_text(),mod.__file__,'exec'),mod.__dict__)
    for job in [j for j in read(E/'time_plan.json')['runs'] if j['method']=='ours']:
        sid=job['id'];root=E/'timing/ours_runs'/f'{sid}_r{job["repeat"]}';p=root/'timing.json';original=root/'history/timing.json.before_export_setup_fix';d=read(p);old=read(original)
        if d['status']=='success':continue
        assert old['failed_phase']=='export' and len(old['stages'])==15
        assert 'NameError: name \'torch\' is not defined' in (root/'history/export.log.before_export_setup_fix').read_text()
        assert all(s['status']=='completed' and s['complete_interval_accounting'] for s in old['stages'])
        assert read(root/'outputs'/sid/'asset/complete.json')['actual_updates']==52999
        for s in old['stages']:assert read(root/'timing'/(s['phase']+'.json'))==s
        d=dict(old);d.update(status='running',resume_controller_sha256=sha(__file__),recovery_started_at=time.time(),infrastructure_failure=dict(record=str(original),sha256=sha(original),classification='export_wrapper_setup_before_timer',explanation='decorated function namespace; inspect.unwrap restores original globals; no export or asset generation was performed by failed setup'),stages=list(old['stages']))
        for key in ['failed_phase','exit_code','failed_elapsed_seconds','finished_at']:d.pop(key,None)
        mod.ROOT=root;mod.ASSET_ROOT=root/'native_assets'
        for phase,library in [('export','grid4d'),('snapshot','4danyone')]:
            assert shutil.disk_usage(E).free>10*1024**3
            proof=root/'timing'/(phase+'.json')
            if proof.exists() and read(proof)['status']=='completed':d['stages'].append(read(proof));continue
            env=mod.environment(os.environ['CUDA_VISIBLE_DEVICES'],library);env.update(HUMAN_RUN_ROOT=str(root),HUMAN_ASSET_ROOT=str(root/'native_assets'))
            for k,sub in [('GRID4D_HASH_BUILD','hash'),('TORCH_EXTENSIONS_DIR','extensions'),('CUDA_CACHE_PATH','cuda')]:env[k]=str(PROD/'cache'/sub)
            cmd=[str(mod.PYTHONS[library]),'-B','-u',str(E/'scripts/time_ours_stage.py'),'--id',sid,'--phase',phase]
            d.update(current_phase=phase,command=cmd,cpu_stat=Path('/proc/stat').read_text().splitlines()[0],diskstats=Path('/proc/diskstats').read_text(),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True));save(p,d)
            with (root/f'{phase}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT);d['child_pid']=child.pid;save(p,d);code=child.wait()
            if proof.exists():d['stages'].append(read(proof))
            if code:
                d.update(status='failed',failed_phase=phase,exit_code=code,failed_elapsed_seconds=sum(s.get('seconds',0) for s in d['stages']),finished_at=time.time());save(p,d);raise RuntimeError(f'{sid} {phase} failed; review before continuing shared implementation')
            save(p,d);print(sid,job['repeat'],phase,'completed',flush=True)
        endpoint=root/'native_assets'/sid/'time_endpoint.json';assert endpoint.exists();d.update(status='success',seconds=sum(s['seconds'] for s in d['stages']),output=str(endpoint),output_sha256=sha(endpoint),independent_reload_status='pending',finished_at=time.time());save(p,d);print(sid,job['repeat'],'success',d['seconds'],flush=True)
if __name__=='__main__':main()
