"""Connect verified system-disk resources and apply upstream's copy_code recipe."""
from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parents[1]
EX = P/'sources/exavatar'
WEIGHTS = Path(_hmc_expand('${HMC_WEIGHT_ROOT}'))
BODY = WEIGHTS/'exavatar_flame_modelscope'
records = []

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def link(source, dest):
    assert source.is_file(), str(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        assert dest.resolve() == source.resolve(), str(dest)
    else:
        dest.symlink_to(source)
    records.append(dict(source=str(source), consumer=str(dest), sha256=sha(source)))

def main():
    # Preserve the full original neutral model: Jacobian's 20-PC derivative is
    # deliberately not reused by ExAvatar, which asks for 100 shape components.
    original = Path(_hmc_expand('${HMC_WORKSPACE}/4danyone/repo/models/body_models/smplx/SMPLX_NEUTRAL.npz'))
    expected = '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'
    assert sha(original) == expected
    neutral = BODY/'smplx/SMPLX_NEUTRAL.npz'
    if not neutral.exists():
        tmp = neutral.with_suffix('.npz.partial');assert not tmp.exists()
        shutil.copyfile(original, tmp);assert sha(tmp)==expected;tmp.replace(neutral)
    assert sha(neutral) == expected
    for root in [EX/'fitting/common/utils/human_model_files',
                 EX/'avatar/common/utils/human_model_files',
                 EX/'fitting/tools/Hand4Whole_RELEASE/common/utils/human_model_files']:
        for f in BODY.rglob('*'):
            if f.is_file() and not f.name.endswith('.partial'):
                rel = f.relative_to(BODY)
                if f.name == 'FLAME_FEMALEl.pkl': rel = rel.with_name('FLAME_FEMALE.pkl')
                link(f, root/rel)
        for gender in ['MALE','FEMALE']:
            link(WEIGHTS/f'modelscope_body_assets_v1/smplx/SMPLX_{gender}.npz', root/f'smplx/SMPLX_{gender}.npz')
    tools = EX/'fitting/tools'
    pairs = [
        (WEIGHTS/'exavatar_drive_v1/deca_model.tar', tools/'DECA/data/deca_model.tar'),
        (BODY/'flame/FLAME_NEUTRAL.pkl', tools/'DECA/data/generic_model.pkl'),
        (BODY/'flame/FLAME_FEMALEl.pkl', tools/'DECA/data/FLAME2020/female_model.pkl'),
        (BODY/'flame/FLAME_MALE.pkl', tools/'DECA/data/FLAME2020/male_model.pkl'),
        (WEIGHTS/'exavatar_drive_v1/snapshot_6.pth.tar', tools/'Hand4Whole_RELEASE/demo/snapshot_6.pth.tar'),
        (WEIGHTS/'depth_anything_v2_vitl.pth', tools/'Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth'),
    ]
    for src, dst in pairs: link(src, dst)
    # Match the upstream copy_code.py explicitly, preserving replaced files.
    copies = [('run_deca.py','DECA/run_deca.py'),
              ('DECA/decalib/deca.py','DECA/decalib/deca.py'),
              ('DECA/decalib/datasets/datasets.py','DECA/decalib/datasets/datasets.py'),
              ('DECA/demos/demo_reconstruct.py','DECA/demos/demo_reconstruct.py'),
              ('run_hand4whole.py','Hand4Whole_RELEASE/demo/run_hand4whole.py'),
              ('mmpose/demo/topdown_demo_with_mmdet.py','mmpose/demo/topdown_demo_with_mmdet.py'),
              ('run_mmpose.py','mmpose/run_mmpose.py'),
              ('run_sam.py','segment-anything/run_sam.py'),
              ('run_depth_anything.py','Depth-Anything-V2/run_depth_anything.py'),
              ('run_colmap.py','COLMAP/run_colmap.py')]
    archive = P/'state/exavatar_upstream_customization_backup';archive.mkdir(exist_ok=True)
    for a,b in copies:
        src=tools/'code_to_copy'/a;dst=tools/b
        if dst.exists() and sha(dst)!=sha(src):
            backup=archive/(sha(dst)+'_'+dst.name)
            if not backup.exists():shutil.copyfile(dst,backup)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        records.append(dict(customization_source=str(src),consumer=str(dst),sha256=sha(dst)))
    (P/'state/exavatar_resource_connections_20261113.json').write_text(json.dumps(dict(status='connected_runtime_validation_pending',files=records),indent=2)+'\n')

if __name__=='__main__':main()
