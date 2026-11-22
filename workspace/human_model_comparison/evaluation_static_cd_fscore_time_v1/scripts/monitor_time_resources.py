"""Low-frequency read-only sampling, separate from measured asset generation."""
import json,time,subprocess,os,shutil,argparse
from pathlib import Path
E=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state-key',default='time_ours');a=ap.parse_args()
    path=E/'timing/resource_samples.jsonl'
    with path.open('a',buffering=1) as f:
        while True:
            row=dict(timestamp=time.time(),cpu_count=os.cpu_count(),load_average=os.getloadavg(),cpu_stat=Path('/proc/stat').read_text().splitlines()[0],diskstats=Path('/proc/diskstats').read_text(),io_pressure=Path('/proc/pressure/io').read_text(),cpu_pressure=Path('/proc/pressure/cpu').read_text(),free_bytes=shutil.disk_usage(E).free)
            for key,command in [('gpu',['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,utilization.memory,power.draw','--format=csv,noheader']),('processes',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])]:
                p=subprocess.run(command,text=True,capture_output=True);row[key]=p.stdout;row[key+'_returncode']=p.returncode
            f.write(json.dumps(row)+'\n')
            state=E/'state'/(a.state_key+'.json')
            if state.exists() and json.loads(state.read_text())['status'] in ['completed','failed']:break
            time.sleep(30)
if __name__=='__main__':main()
