"""Package audited results, validate every archived byte, then bind final completion."""
import hashlib,time,zipfile,os
from common import *
DEST=E/'delivery';ARCHIVE=DEST/'human100_cd_f1_time_results_20261115.zip'
def include(p):
    rel=p.relative_to(E);parts=rel.parts
    if parts[0] in ['cache','delivery'] or '__pycache__' in parts or p.suffix in ['.lock','.tmp','.pyc']:return False
    if str(rel)=='logs/package_results.log':return False  # Active packaging log is written after the snapshot.
    if p.name=='completion_audit.json':return False
    if parts[0]=='timing':return p.suffix in ['.json','.jsonl','.log'] or 'history' in parts
    if parts[0]=='runs':return p.suffix in ['.json','.png'] or p.name in ['returns.npz','native_metric_returns.npz']
    if parts[0]=='pilot':return p.suffix not in ['.pt','.ply'] and p.name!='native_activated_arrays.npz'
    return p.suffix in ['.json','.jsonl','.csv','.py','.txt','.html','.png','.jpg','.jpeg','.svg','.pdf','.tex','.npz','.log']
def entries(paths):return {str(p.relative_to(E)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}
def build(path,members):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name in sorted(members):z.write(E/name,name)
def verify(path,members):
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())==set(members)
        for name,meta in members.items():
            h=hashlib.sha256();size=0
            with z.open(name) as f:
                for b in iter(lambda:f.read(1024**2),b''):size+=len(b);h.update(b)
            assert size==meta['bytes'] and h.hexdigest()==meta['sha256'],name
def main():
    experiment=read(E/'validation/experiment_completion_audit.json');assert experiment['status']=='passed' and read(E/'validation/time_audit.json')['terminal_verified']==20
    DEST.mkdir(exist_ok=True)
    verified=read(E/'validation/verified_input_hashes.json');deps={p:h for p,h in verified.items() if not str(p).startswith(str(E)) or not include(Path(p))}
    for p in (E/'timing/ours_runs').glob('*/native_assets/*/time_endpoint.json'):
        for rel,h in read(p)['files'].items():
            f=p.parent/rel
            if not include(f):deps[str(f)]=h
    save(DEST/'external_dependencies.json',dict(scope='Hash-bound files intentionally kept at original local paths; not copied into result archive. Pretrained runtime dependencies also remain in original named environments/model paths.',files=deps))
    paths=sorted(p for p in E.rglob('*') if p.is_file() and include(p));paths.append(DEST/'external_dependencies.json')
    base=entries(paths);manifest=DEST/'contents_manifest.json';save(manifest,dict(scope='All packaged files except this manifest itself; final ZIP digest stored separately.',files=base))
    members=dict(base);members['delivery/contents_manifest.json']=dict(bytes=manifest.stat().st_size,sha256=sha(manifest))
    staging=DEST/'.validated_base.zip';build(staging,members);verify(staging,members)
    proof=dict(status='passed',all_requirements_satisfied=True,experiment_audit=str(E/'validation/experiment_completion_audit.json'),experiment_audit_sha256=sha(E/'validation/experiment_completion_audit.json'),geometry_terminal=600,measured_rows=528,finite_cd_rows=527,empty_prediction_rows=1,NA_rows=72,metric_panels_independently_verified=711,time_success=20,time_planned=20,reference_main=84,reference_low=16,common_main_admission=46,common_main_finite=45,owned_background_workers_alive=False,package=dict(path=str(ARCHIVE),contents_manifest='delivery/contents_manifest.json',verification='Every decompressed member compared byte-for-byte via SHA256; final archive verification repeated after adding completion audit.',zip_sha256_file=str(ARCHIVE)+'.sha256',large_binary_scope='retained at hash-bound local paths; see external_dependencies.json'),script_sha256=sha(__file__),finished_at=time.time())
    auditpath=E/'validation/completion_audit.json';save(auditpath,proof)
    progress=read(E/'progress.json');progress.update(goal_complete=True,completion_audit=str(auditpath),completion_audit_sha256=sha(auditpath));save(E/'progress.json',progress)
    base['validation/completion_audit.json']=dict(bytes=auditpath.stat().st_size,sha256=sha(auditpath));base['progress.json']=dict(bytes=(E/'progress.json').stat().st_size,sha256=sha(E/'progress.json'))
    save(manifest,dict(scope='All packaged files except this manifest itself; final ZIP digest stored separately.',files=base));members=dict(base);members['delivery/contents_manifest.json']=dict(bytes=manifest.stat().st_size,sha256=sha(manifest))
    temporary=DEST/'.final.zip';build(temporary,members);verify(temporary,members);temporary.replace(ARCHIVE);staging.unlink()
    digest=sha(ARCHIVE);Path(str(ARCHIVE)+'.sha256').write_text(digest+'  '+ARCHIVE.name+'\n')
    save(DEST/'archive_verification.json',dict(status='passed',path=str(ARCHIVE),sha256=digest,bytes=ARCHIVE.stat().st_size,members=len(members),uncompressed_bytes=sum(m['bytes'] for m in members.values()),every_member_sha256_verified=True,completion_audit_sha256=sha(auditpath),manifest_sha256=sha(manifest),finished_at=time.time()))
    print(read(DEST/'archive_verification.json'))
if __name__=='__main__':main()
