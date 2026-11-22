"""Remove only 32 completed alpha/A optimizer recovery files, preserving finals.

User-authorized cleanup; the eight native recovery files used by the recorded
budget-control protocol are explicitly excluded. Exact deletions and retained
final asset/metric hashes are audited before and after the operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

H=Path(__file__).resolve().parents[1]
BASE=H/'runs/backend_ablation_v1'
OUT=H/'state/completed_ablation_optimizer_cleanup_20261115.json'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for data in iter(lambda:f.read(16*1024**2),b''):h.update(data)
    return h.hexdigest()
def save(d):
    t=OUT.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(OUT)

def main(execute):
    assert not OUT.exists(),'Cleanup already audited; inspect existing result instead of repeating'
    source=H/'state/completed_ablation_recovery_cleanup_candidates_20261115.json';rows=read(source)['rows']
    protocols=list(BASE.glob('*/rgb_continuation_53k/protocol.json'));assert len(protocols)==8
    keep={Path(read(p)['native_recovery']).resolve()for p in protocols};assert len(keep)==8
    selected=[r for r in rows if Path(r['path']).resolve()not in keep];assert len(selected)==32
    live=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            scripts=[x.decode(errors='replace')for x in args[1:]if x.endswith(b'.py')]
            if any('backend_ablation' in x or 'reconstruction_rgb_control' in x for x in scripts):live.append(dict(pid=int(p.name),scripts=scripts))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    assert not live,live
    preserved={str(p):sha(p)for p in protocols};deleted=[]
    for root in [H/'reports/backend_ablation_v1',H/'reports/backend_ablation_batch_v2']:
        for p in root.rglob('*'):
            if p.is_file()and p.suffix in ['.json','.csv','.md','.tex']:preserved[str(p)]=sha(p)
    for row in selected:
        p=Path(row['path']);assert p.is_relative_to(BASE)and p.name=='recovery.pt'and p.parent.name in ['alpha','asset']and not p.is_symlink()
        stat=p.stat();assert stat.st_size==row['bytes']and stat.st_nlink==1
        complete=p.parent/'complete.json';assert read(complete)==row['complete']and row['complete']['state']in ['complete_A','alpha_complete']
        finals=[*row['point_clouds'],*row['deforms']];assert len(row['point_clouds'])==len(row['deforms'])==1
        for f in [complete,*map(Path,finals)]:assert f.is_file();preserved[str(f)]=sha(f)
        deleted.append(dict(path=str(p),bytes=stat.st_size,sha256=sha(p),mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,
                            complete=str(complete),final_files=finals,reason='Completed alpha/A training; optimizer resume state unused by exported final PLY/deform assets or metric manifests'))
        print('verified',len(deleted),p,flush=True)
    kept={str(p):dict(bytes=p.stat().st_size,sha256=sha(p))for p in keep}
    audit=dict(status='verified_before_deletion',source_candidates=str(source),source_candidates_sha256=sha(source),script_sha256=sha(__file__),
        planned_files=deleted,preserved_final_and_metric_sha256=preserved,preserved_protocol_native_recovery=kept,
        dependency_audit=dict(live_ablation_processes=[],scope='Both projects configs/state/exports/reports plus human runs JSON/CSV/YAML',
            only_recovery_manifest_references=[str(p)for p in protocols],evaluation_loader=str(H/'scripts/evaluate_backend_ablation.py'),evaluation_loader_sha256=sha(H/'scripts/evaluate_backend_ablation.py'),
            evaluator_reads='config, asset_camera_data, snapshots, final point_cloud.ply and deform.pth; recovery is optimizer state read only by training restore/resume'),
        disk_free_before=shutil.disk_usage(H).free,started_at=time.time(),deleted=[])
    save(audit)
    if not execute:return
    for record in deleted:
        p=Path(record['path']);stat=p.stat();assert stat.st_ino==record['inode']and stat.st_mtime_ns==record['mtime_ns']and stat.st_size==record['bytes']
        p.unlink();audit['deleted'].append(record['path']);save(audit)
    for p,digest in preserved.items():assert sha(p)==digest,p
    for p,binding in kept.items():assert sha(p)==binding['sha256'],p
    audit.update(status='completed',deleted_files=len(audit['deleted']),deleted_bytes=sum(r['bytes']for r in deleted),disk_free_after=shutil.disk_usage(H).free,
        preserved_final_and_metric_hashes_unchanged=True,preserved_eight_native_recovery_hashes_unchanged=True,finished_at=time.time())
    save(audit);print(json.dumps(dict(status='completed',deleted_files=32,deleted_GiB=audit['deleted_bytes']/2**30,audit=str(OUT))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.execute)
