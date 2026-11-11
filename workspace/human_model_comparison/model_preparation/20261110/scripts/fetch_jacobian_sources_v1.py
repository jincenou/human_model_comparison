"""Fetch the six upstream-pinned sources, without executing repository scripts."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

P = Path(__file__).resolve().parents[1]
REPO = P/'sources/jacobianavatar'
STATE = P/'state/jacobian_sources_v1.json'


def main():
    lock = (P/'state/jacobian_sources_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not STATE.exists(), 'Inspect existing attempt before launching another.'
    source = REPO/'scripts/usage/fetch_libs.sh'
    entries = re.findall(r'^clone_at\s+(\w+)\s+(https://github.com/[\w.-]+/[\w.-]+\.git)\s+([a-f0-9]{40})', source.read_text(), re.M)
    assert len(entries) == 6
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_LFS_SKIP_SMUDGE='1')
    state = dict(status='fetching', pid=os.getpid(), started_at=time.time(), jobs=[],
        official_fetch_script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method_runtime_validated=False, subprocess_timeout_seconds=600)
    def save():
        temp=STATE.with_suffix('.tmp'); temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(STATE)
    def git(path,*args,capture=False):
        cmd=['git','-c','http.proxy=','-c','https.proxy=','-C',str(path),*args]
        if capture:
            return subprocess.check_output(cmd,env=env,text=True,timeout=600).strip()
        child=subprocess.Popen(cmd,env=env); state['active_child_pid']=child.pid; save()
        try: code=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            child.terminate(); child.wait(); raise
        assert code==0, (cmd,code)
    save()
    for name,url,revision in entries:
        row=dict(name=name,url=url,revision=revision,status='fetching'); state['jobs'].append(row); save()
        try:
            assert shutil.disk_usage(P).free > 12*1024**3
            dest=REPO/'libs'/name
            dest.mkdir(exist_ok=False)
            git(dest,'init','-q'); git(dest,'remote','add','origin',url)
            git(dest,'fetch','--depth','1','origin',revision)
            git(dest,'checkout','--detach','FETCH_HEAD')
            head=git(dest,'rev-parse','HEAD',capture=True); assert head==revision
            modules=dest/'.gitmodules'
            row.update(status='source_checkout_verified',path=str(dest),actual_revision=head,
                submodule_configuration=modules.read_text() if modules.exists() else None,
                submodules_pending=modules.exists())
        except Exception as exc:
            row.update(status='failed',reason=repr(exc))
        save()
    state.update(status='sources_fetched_submodules_pending' if all(j['status']=='source_checkout_verified' for j in state['jobs']) else 'partial_failure', finished_at=time.time())
    state.pop('active_child_pid',None);save()


if __name__=='__main__':
    main()
