"""Reproduce the pinned upstream 10-shape/10-expression conversion on local assets."""
from human_asset_benchmark.paths import expand as _hmc_expand
import hashlib
import json
import pickle
from pathlib import Path
import shutil

import numpy as np
import torch
import smplx

P=Path(__file__).resolve().parents[1]
DEST=Path(_hmc_expand('${HMC_WEIGHT_ROOT}/jacobian_smplx_v1'))
SOURCES={
    'npz':(Path(_hmc_expand('${HMC_WORKSPACE}/4danyone/repo/models/body_models/smplx/SMPLX_NEUTRAL.npz')),'376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'),
    'pkl':(Path(_hmc_expand('${HMC_WORKSPACE}/idol/weights/SMPLX_NEUTRAL.pkl')),'4b1dd056fcf6f6d5ece7ba54f7b432cf6877ca8624e6de6ccbf9815d24326965')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert shutil.disk_usage(DEST.parent).free>11*1024**3
    assert not DEST.exists()
    for path,digest in SOURCES.values():assert sha(path)==digest
    original={'npz':dict(np.load(SOURCES['npz'][0],allow_pickle=True)),
              'pkl':pickle.load(SOURCES['pkl'][0].open('rb'),encoding='latin1')}
    for key in ('v_template','shapedirs','posedirs','weights','f','kintree_table','J_regressor'):
        assert np.array_equal(original['npz'][key],original['pkl'][key]),key
    DEST.mkdir()
    outputs={}
    for ext,data in original.items():
        assert data['shapedirs'].shape==(10475,3,400)
        slim=dict(data);slim.pop('vt',None);slim.pop('ft',None)
        slim['shapedirs']=np.concatenate([data['shapedirs'][...,:10],data['shapedirs'][...,300:310]],axis=-1)
        path=DEST/f'SMPLX_NEUTRAL.{ext}'
        if ext=='npz':np.savez(path,**slim)
        else:
            with path.open('wb') as f:pickle.dump(slim,f,protocol=2)
        outputs[ext]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'shapedirs_shape':list(slim['shapedirs'].shape)}
    torch.set_num_threads(4)
    params=dict(gender='neutral',num_betas=10,num_expression_coeffs=10,use_pca=False,flat_hand_mean=True)
    full=smplx.SMPLX(model_path=str(SOURCES['npz'][0]),ext='npz',**params)
    variants={ext:smplx.SMPLX(model_path=str(DEST/f'SMPLX_NEUTRAL.{ext}'),ext=ext,**params) for ext in ('npz','pkl')}
    torch.manual_seed(641)
    checks=[]
    for case in ['zero','nonzero']:
        inputs={k:torch.zeros(1,n) if case=='zero' else torch.randn(1,n)*0.07 for k,n in
            [('betas',10),('expression',10),('body_pose',63),('global_orient',3),('jaw_pose',3),('left_hand_pose',45),('right_hand_pose',45)]}
        with torch.no_grad():
            a=full(**inputs)
            for ext,model in variants.items():
                b=model(**inputs);v=float((a.vertices-b.vertices).abs().max());j=float((a.joints-b.joints).abs().max())
                assert v<1e-6 and j<1e-6,(case,ext,v,j)
                checks.append(dict(case=case,extension=ext,vertex_max_error=v,joint_max_error=j))
    resource=P/'sources/jacobianavatar/resource/smplx';resource.mkdir(parents=True,exist_ok=True)
    for ext,row in outputs.items():
        link=resource/f'SMPLX_NEUTRAL.{ext}';assert not link.exists() and not link.is_symlink();link.symlink_to(row['path'])
    for path,digest in SOURCES.values():assert sha(path)==digest
    record=dict(status='converted_and_forward_equivalence_passed',scope='Neutral SMPL-X only, not ExAvatar male model or complete Jacobian readiness',
        sources={ext:{'path':str(path),'sha256':digest} for ext,(path,digest) in SOURCES.items()},outputs=outputs,
        checks=checks,script_sha256=sha(__file__),official_conversion_source=str(P/'sources/jacobianavatar/scripts/usage/fetch_assets.sh'),
        official_conversion_source_sha256=sha(P/'sources/jacobianavatar/scripts/usage/fetch_assets.sh'),torch_version=torch.__version__,
        validation_device='cpu',method_runtime_validated=False)
    (P/'state/jacobian_smplx_conversion_v1.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
