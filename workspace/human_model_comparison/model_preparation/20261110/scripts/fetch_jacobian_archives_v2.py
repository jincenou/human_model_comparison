"""Recover terminal Git transport failures using official commit archives."""
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
import urllib.request

P=Path(__file__).resolve().parents[1]
STATE=P/'state/jacobian_archive_recovery_v2.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    lock=(P/'state/jacobian_archive_recovery_v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists()
    state=dict(status='recovering_terminal_git_failures',pid=os.getpid(),started_at=time.time(),jobs=[])
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    archives=P/'sources/jacobian_commit_archives_v2';archives.mkdir(exist_ok=False)
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
    save()
    while True:
        upstream=json.loads((P/'state/jacobian_sources_v1.json').read_text())
        done={j['name'] for j in state['jobs']}
        for job in upstream['jobs']:
            if job['name'] in done or job['status']!='failed':continue
            row=dict(name=job['name'],status='downloading',revision=job['revision']);state['jobs'].append(row);save()
            try:
                assert shutil.disk_usage(P).free>12*1024**3
                repo=job['url'][len('https://github.com/'):-4]
                url=f"https://codeload.github.com/{repo}/tar.gz/{job['revision']}"
                archive=archives/f"{job['name']}-{job['revision']}.tar.gz";temp=archive.with_suffix('.part')
                size=0
                with opener.open(url,timeout=90) as response,temp.open('xb') as out:
                    while chunk:=response.read(4*1024**2):
                        assert shutil.disk_usage(P).free>11*1024**3
                        size+=len(chunk);assert size<2*1024**3
                        out.write(chunk)
                temp.replace(archive)
                dest=P/'sources/jacobianavatar/libs'/job['name']
                assert dest.exists() and all(x.name=='.git' for x in dest.iterdir())
                expected=repo.split('/')[-1]+'-'+job['revision']
                with tarfile.open(archive) as tar:
                    members=tar.getmembers()
                    assert sum(m.size for m in members)<3*1024**3
                    for member in members:
                        parts=PurePosixPath(member.name).parts
                        assert parts and parts[0]==expected and '..' not in parts
                        assert member.isdir() or member.isfile(), (member.name,'nonregular member needs explicit inspection')
                    for member in members:
                        relative=PurePosixPath(member.name).parts[1:]
                        if not relative:continue
                        path=dest.joinpath(*relative)
                        if member.isdir():path.mkdir(parents=True,exist_ok=True)
                        else:
                            path.parent.mkdir(parents=True,exist_ok=True)
                            with tar.extractfile(member) as src,path.open('xb') as out:shutil.copyfileobj(src,out)
                            path.chmod(member.mode & 0o777)
                files={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file() and '.git' not in f.relative_to(dest).parts}
                modules=dest/'.gitmodules'
                provenance=dict(status='official_commit_archive_extracted',url=url,revision=job['revision'],archive_sha256=sha(archive),
                    files_sha256=files,source_tree_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
                    original_git_failure=job,submodule_configuration=modules.read_text() if modules.exists() else None,
                    method_runtime_validated=False)
                (dest/'_source_archive_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
                row.update(status='official_commit_archive_extracted',path=str(dest),archive_sha256=provenance['archive_sha256'],files=len(files),submodules_pending=modules.exists())
            except Exception as exc:row.update(status='failed',reason=repr(exc))
            save()
        if upstream['status']!='fetching':break
        time.sleep(10)
    state.update(status='finished' if all(j['status']=='official_commit_archive_extracted' for j in state['jobs']) else 'finished_with_failures',finished_at=time.time());save()


if __name__=='__main__':main()
