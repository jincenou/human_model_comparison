"""Fixed five-run Ours schedule, full unchanged native optimization budgets."""
import os,subprocess,sys,time,shutil,importlib.util
from common import *
CODE=H.parent/'human';PRODUCTION=H.parent/'asset_work/human_nuscenes100_20261026'
def main():
    plan=read(E/'time_plan.json');registry=read(PRODUCTION/'config/cases.json')
    module=__import__('types').ModuleType('human_native_common');module.__file__=str(CODE/'common.py');exec(compile((CODE/'common.py').read_text(),str(CODE/'common.py'),'exec'),module.__dict__)
    phases=[('native_input','4danyone'),('interpolate','4danyone'),('segment','4danyone-sam2'),('motion','4danyone'),('condition','4danyone'),('frontend','4danyone')]+[('masks','4danyone-sam2',i) for i in range(4)]+[('hull','4danyone-ftgs'),('configure','4danyone-ftgs'),('native','grid4d'),('alpha','grid4d'),('A','grid4d'),('export','grid4d'),('snapshot','4danyone')]
    for job in [j for j in plan['runs'] if j['method']=='ours']:
        sid=job['id'];root=E/'timing/ours_runs'/f'{sid}_r{job["repeat"]}';rp=root/'timing.json'
        if rp.exists() and read(rp)['status'] in ['success','failed']:continue
        root.mkdir(parents=True,exist_ok=True);(root/'config').mkdir(exist_ok=True)
        if not (root/'config/cases.json').exists():
            spec=read(PRODUCTION/'config/cases.json')[sid]
            # Redirect only owned production paths; original raw RGB stays fixed.
            spec=__import__('json').loads(__import__('json').dumps(spec).replace(str(PRODUCTION),str(root)))
            save(root/'config/cases.json',{sid:spec});save(root/'selection.json',dict(items=[dict(registry[sid]['selection'],id=sid)]));shutil.copy2(CODE/'config/export.json',root/'config/export.json')
            save(root/'input_binding.json',dict(plan_sha256=sha(E/'time_plan.json'),original_registry_sha256=sha(PRODUCTION/'config/cases.json'),selection_sha256=SELECTION_SHA,raw_files={r['image_path']:sha(r['image_path']) for r in registry[sid]['selection']['observations']}))
        module.ROOT=root;module.ASSET_ROOT=root/'native_assets';record=dict(job,status='running',plan_sha256=sha(E/'time_plan.json'),started_at=time.time(),stages=[],gpu=os.environ['CUDA_VISIBLE_DEVICES'],device='NVIDIA GeForce RTX 4090',timing='sum of synchronized active intervals between exact named pretrained-load boundaries; full preprocessing/training/necessary I/O/export; not online latency',script_sha256=sha(__file__),stage_script_sha256=sha(E/'scripts/time_ours_stage.py'))
        save(rp,record)
        for phase in phases:
            name,library,*group=phase;tag=name+(str(group[0]) if group else '');proof=root/'timing'/f'{tag}.json'
            if proof.exists() and read(proof)['status']=='completed':record['stages'].append(read(proof));continue
            assert shutil.disk_usage(E).free>10*1024**3
            ev=module.environment(os.environ['CUDA_VISIBLE_DEVICES'],library);ev.update(HUMAN_RUN_ROOT=str(root),HUMAN_ASSET_ROOT=str(root/'native_assets'))
            # Reuse production extension caches; do not upgrade dependencies.
            for k,sub in [('GRID4D_HASH_BUILD','hash'),('TORCH_EXTENSIONS_DIR','extensions'),('CUDA_CACHE_PATH','cuda')]:ev[k]=str(PRODUCTION/'cache'/sub)
            command=[str(module.PYTHONS[library]),'-B','-u',str(E/'scripts/time_ours_stage.py'),'--id',sid,'--phase',name]
            if group:command+=['--group',str(group[0])]
            record.update(current_phase=tag,command=command,cpu_stat=Path('/proc/stat').read_text().splitlines()[0],diskstats=Path('/proc/diskstats').read_text(),gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True));save(rp,record)
            with (root/f'{tag}.log').open('a') as log:
                child=subprocess.Popen(command,cwd=CODE,env=ev,stdout=log,stderr=subprocess.STDOUT);record['child_pid']=child.pid;save(rp,record);code=child.wait()
            if proof.exists():record['stages'].append(read(proof))
            if code:
                record.update(status='failed',failed_phase=tag,exit_code=code,failed_elapsed_seconds=sum(s.get('seconds',0) for s in record['stages']));break
            save(rp,record);print(sid,job['repeat'],tag,'completed',flush=True)
        else:
            endpoint=root/'native_assets'/sid/'time_endpoint.json';assert endpoint.exists()
            record.update(status='success',seconds=sum(s['seconds'] for s in record['stages']),output=str(endpoint),output_sha256=sha(endpoint),independent_reload_status='pending')
        record['finished_at']=time.time();save(rp,record);print(sid,job['repeat'],record['status'],record.get('seconds'),flush=True)

if __name__=='__main__':main()
