"""One pinned generation worker exits naturally, then this GPU joins source scoring."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import evaluate_human_static_two_models as src

def start_ticks(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else int(fields[19])
    except FileNotFoundError:return None

def main(gpu,pid,ticks):
    out=src.OUT/'continuations';out.mkdir(exist_ok=True)
    with (out/f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='waiting_pinned_generation_worker',pid=os.getpid(),gpu=gpu,generation_pid=pid,generation_start_ticks=ticks,started_at=time.time());src.save(out/f'gpu{gpu}.json',state)
        while start_ticks(pid)==ticks:time.sleep(15)
        generation=src.read(src.GEN/'workers'/f'gpu{gpu}.json')
        if generation.get('pid')!=pid or generation['status']!='no_eligible_pending_jobs':
            src.save(out/f'gpu{gpu}.json',dict(state,status='generation_exit_needs_review',generation_state=generation,finished_at=time.time()));return
        cmd=[src.PY['control'],'-B','-u',str(src.H/'scripts/run_human_static_two_models_staged.py'),'--gpu',str(gpu),'--batch-size','4']
        with (src.OUT/f'batch_gpu{gpu}_staged.log').open('a') as f:child=subprocess.Popen(cmd,cwd=src.H,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        src.save(src.OUT/f'batch_gpu{gpu}_staged_launcher.json',dict(pid=child.pid,command=cmd,started_at=time.time(),after_generation_pid=pid))
        src.save(out/f'gpu{gpu}.json',dict(state,status='source_started_after_natural_generation_exit',source_pid=child.pid,generation_state=generation,started_source_at=time.time()))
        code=child.wait();src.save(out/f'gpu{gpu}.json',dict(state,status='completed' if code==0 else 'source_execution_failed',source_pid=child.pid,exit_code=code,finished_at=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--generation-pid',type=int,required=True);p.add_argument('--generation-start-ticks',type=int,required=True);a=p.parse_args();main(a.gpu,a.generation_pid,a.generation_start_ticks)
