"""Retry the exact official MHR release, requiring its published size and SHA256."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import fcntl,hashlib,json,os,shutil,time,zipfile
import requests
P=Path(__file__).resolve().parents[1]
STATE=P/'state/mhr_release_download_v2.json'
SIZE=198943157
DIGEST='e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94'
def main():
    lock=STATE.with_suffix('.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not STATE.exists()
    dest=Path(_hmc_expand('${HMC_WEIGHT_ROOT}/mhr_v1_0_1'));dest.mkdir(exist_ok=True)
    archive=dest/'assets.zip';assert not archive.exists()
    part=archive.with_suffix('.zip.part');assert not part.exists()
    s=dict(status='connecting_official_release',pid=os.getpid(),started_at=time.time(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source='https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip',expected_bytes=SIZE,expected_sha256=DIGEST,bytes_received=0,method_runtime_validated=False)
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(STATE)
    try:
        save();session=requests.Session();session.trust_env=False
        with session.get(s['source'],stream=True,timeout=(20,120)) as response:
            response.raise_for_status()
            assert response.url.startswith(('https://release-assets.githubusercontent.com/','https://github.com/'))
            s['status']='downloading';save();digest=hashlib.sha256();last=0
            with part.open('xb') as f:
                for block in response.iter_content(1024**2):
                    assert shutil.disk_usage(dest).free>12*1024**3
                    if not block:continue
                    f.write(block);digest.update(block);s['bytes_received']+=len(block)
                    assert s['bytes_received']<=SIZE
                    if time.time()-last>5:s['updated_at']=time.time();save();last=time.time()
            assert s['bytes_received']==SIZE and digest.hexdigest()==DIGEST
        part.replace(archive);s.update(status='verified_extracting',archive=str(archive),sha256=DIGEST);save()
        names=['compact_v6_1.model','corrective_activation.npz','corrective_blendshapes_lod1.npz','corrective_blendshapes_lod3.npz','lod1.fbx','lod3.fbx']
        extracted=[]
        with zipfile.ZipFile(archive) as z:
            bad=z.testzip();assert bad is None,bad
            for name in names:
                members=[i for i in z.infolist() if Path(i.filename).name==name and not i.is_dir()]
                assert len(members)==1,(name,len(members))
                info=members[0];assert shutil.disk_usage(dest).free-info.file_size>12*1024**3
                out=dest/name;assert not out.exists()
                digest=hashlib.sha256()
                with z.open(info) as src,out.open('xb') as dst:
                    for block in iter(lambda:src.read(1024**2),b''):dst.write(block);digest.update(block)
                extracted.append(dict(name=name,path=str(out),bytes=out.stat().st_size,sha256=digest.hexdigest(),archive_member=info.filename))
        s.update(status='official_digest_and_six_assets_verified',files=extracted)
    except BaseException as exc:
        s.update(status='failed',reason=repr(exc));raise
    finally:s['finished_at']=time.time();save()
if __name__=='__main__':main()
