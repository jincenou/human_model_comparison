"""Serial fair-window direct-method attempts; failures retain planned identities."""
import os,subprocess,sys,time
from common import *
state=dict(status='running',pid=os.getpid(),started_at=time.time(),stages=[])
for method in ['trellis_original','ah','trellis2']:
    sp=E/'state'/f'time_{method}.json'
    if sp.exists() and read(sp)['status']=='completed':continue
    command=[sys.executable,'-B','-u',str(E/'scripts/run_gpu.py'),'--method',method,'--gpu','4','--key',f'time_{method}',str(E/'scripts/time_direct.py'),'--method',method]
    state['current_method']=method;save(E/'state/time_sequence.json',state)
    code=subprocess.call(command);state['stages'].append(dict(method=method,exit_code=code,finished_at=time.time()));save(E/'state/time_sequence.json',state)
state.update(status='completed',finished_at=time.time());save(E/'state/time_sequence.json',state)
