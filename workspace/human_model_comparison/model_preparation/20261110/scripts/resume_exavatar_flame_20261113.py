"""Resume only the pinned paused FLAME transfer; preserve historical state."""
import fcntl,json,os,shutil,time
from pathlib import Path
import requests
from download_exavatar_flame_models import P,DEST,INVENTORY,FILES,sha
STATE=P/'state/exavatar_flame_resume_20261113.json'

def main():
    with (P/'state/exavatar_flame_download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not STATE.exists()
        old=P/'state/exavatar_flame_download.json'
        assert json.loads(old.read_text())['status']=='paused_by_user_priority'
        meta={r['Path']:r for r in json.loads(INVENTORY.read_text())['Data']['Files']}
        state=dict(status='running',pid=os.getpid(),started_at=time.time(),previous_state=str(old),previous_sha256=sha(old),files=[],runtime_validated=False)
        def save():
            tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
        session=requests.Session();session.trust_env=False;save()
        try:
            for name in FILES:
                m=meta[name];dest=DEST/Path(name).relative_to('human_model_files');partial=dest.with_suffix(dest.suffix+'.partial')
                row=dict(source=name,revision=m['Revision'],path=str(dest),expected_bytes=m['Size'],expected_sha256=m['Sha256'],status='checking');state['files'].append(row);save()
                if not dest.exists():
                    offset=partial.stat().st_size if partial.exists() else 0
                    assert offset<=m['Size']
                    assert shutil.disk_usage(DEST).free-(m['Size']-offset)>4*1024**3
                    response=session.get('https://modelscope.cn/api/v1/models/Damo_XR_Lab/LHMPP-Prior/repo',params={'Revision':m['Revision'],'FilePath':name},headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120))
                    response.raise_for_status()
                    if offset:
                        assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-'), 'Server did not acknowledge the resume range; partial preserved'
                    row.update(status='downloading',resumed_from=offset);save();last=time.monotonic()
                    with partial.open('ab' if offset else 'xb') as f:
                        for chunk in response.iter_content(4*1024**2):
                            if not chunk:continue
                            assert shutil.disk_usage(DEST).free>3*1024**3
                            f.write(chunk)
                            if time.monotonic()-last>10:
                                row['received_bytes']=f.tell();save();last=time.monotonic()
                        f.flush();os.fsync(f.fileno())
                    response.close()
                    assert partial.stat().st_size==m['Size'] and sha(partial)==m['Sha256']
                    partial.replace(dest)
                assert dest.stat().st_size==m['Size'] and sha(dest)==m['Sha256']
                row.update(status='downloaded_hash_verified',sha256=m['Sha256']);save();print(name,row['status'],flush=True)
            state['status']='downloaded_hash_verified_runtime_pending'
        except Exception as e:
            state.update(status='failed_retained',error=repr(e));raise
        finally:
            state['finished_at']=time.time();save()
if __name__=='__main__':main()
