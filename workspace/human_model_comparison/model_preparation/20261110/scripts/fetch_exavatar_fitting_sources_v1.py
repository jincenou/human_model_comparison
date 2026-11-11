"""Fetch official fitting-tool sources at recorded commits without installing them."""
from pathlib import Path
import fcntl,hashlib,json,os,shutil,tarfile,time
import requests
P=Path(__file__).resolve().parents[1]
STATE=P/'state/exavatar_fitting_sources_v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    lock=STATE.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not STATE.exists()
    state=dict(status='running',pid=os.getpid(),started_at=time.time(),script_sha256=sha(__file__),jobs=[],method_runtime_validated=False)
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
    archive_root=P/'sources/exavatar_fitting_archives_v1';archive_root.mkdir(exist_ok=False)
    for repo,folder in [('yfeng95/DECA','DECA'),('mks0601/Hand4Whole_RELEASE','Hand4Whole_RELEASE'),('facebookresearch/segment-anything','segment-anything'),('DepthAnything/Depth-Anything-V2','Depth-Anything-V2')]:
        probe=P/'reports/exavatar_tool_sources_probe_v1'/(repo.replace('/','_')+'.json')
        data=json.loads(probe.read_text());assert data['repository']==repo;rev=data['revision'];assert len(rev)==40
        target=P/'sources/exavatar/fitting/tools'/folder;assert not target.exists()
        job=dict(repository=repo,revision=rev,target=str(target),status='downloading',probe_sha256=sha(probe));state['jobs'].append(job);save()
        try:
            assert shutil.disk_usage(P).free>13*1024**3
            archive=archive_root/(folder+'-'+rev+'.tar.gz')
            session=requests.Session();session.trust_env=False
            with session.get(f'https://codeload.github.com/{repo}/tar.gz/{rev}',stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                with archive.open('xb') as f:
                    for block in response.iter_content(1024**2):
                        f.write(block);assert f.tell()<300*1024**2
            job.update(archive=str(archive),archive_sha256=sha(archive),status='extracting');save()
            with tarfile.open(archive) as tf:
                members=tf.getmembers();root=members[0].name.split('/')[0]
                assert root.endswith(rev)
                for member in members:
                    path=Path(member.name)
                    assert not path.is_absolute() and '..' not in path.parts
                    assert not member.issym() and not member.islnk() and (member.isfile() or member.isdir()),member.name
                temp=archive_root/(folder+'_extract');temp.mkdir();tf.extractall(temp)
            (temp/root).rename(target);temp.rmdir()
            job.update(status='official_pinned_source_available',files=sum(1 for f in target.rglob('*') if f.is_file()),finished_at=time.time())
        except Exception as exc:job.update(status='failed',reason=repr(exc),finished_at=time.time())
        save()
    state.update(status='finished' if all(j['status']=='official_pinned_source_available' for j in state['jobs']) else 'finished_with_failures',finished_at=time.time());save()
if __name__=='__main__':main()
