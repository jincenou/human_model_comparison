"""Remove legacy Chumpy containers without altering model arrays or shape bases.

Run conversion in the existing ExAvatar CPU environment, then validate loading
and forward/backward in Jacobian's own environment before linking consumers.
"""
from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,hashlib,inspect,json,pickle,shutil
from pathlib import Path
import numpy as np
import torch
import smplx
P=Path(__file__).resolve().parents[1]
SOURCE=Path(_hmc_expand('${HMC_WEIGHT_ROOT}/modelscope_body_assets_v1/smpl/SMPL_NEUTRAL.pkl'))
DIGEST='4924f235e63f7c5d5b690acedf736419c2edb846a2d69fc0956169615fa75688'
DEST=Path(_hmc_expand('${HMC_WEIGHT_ROOT}/jacobian_smpl_neutral_v1'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(data):
    return {k:np.asarray(v.r) if type(v).__module__.startswith('chumpy.') else v for k,v in data.items()}
def cases(model):
    result={};checks=[];rng=np.random.default_rng(713);torch.set_num_threads(4)
    for case in ['zero','nonzero']:
        ins={k:torch.tensor(np.zeros((1,n),np.float32) if case=='zero' else rng.normal(0,.06,(1,n)).astype(np.float32),requires_grad=True)
             for k,n in [('betas',10),('body_pose',69),('global_orient',3),('transl',3)]}
        out=model(**ins);loss=out.vertices.square().mean();loss.backward()
        assert out.vertices.shape==(1,6890,3) and torch.isfinite(out.vertices).all()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in ins.values())
        result[case+'_vertices']=out.vertices.detach().numpy();result[case+'_joints']=out.joints.detach().numpy()
        for k,v in ins.items():result[case+'_'+k+'_grad']=v.grad.numpy()
        checks.append(dict(case=case,vertices=list(out.vertices.shape),joints=list(out.joints.shape),finite_forward_and_backward=True))
    return result,checks

def convert():
    assert sha(SOURCE)==DIGEST and shutil.disk_usage(DEST.parent).free>11*1024**3
    assert not DEST.exists()
    for k,v in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('str',str),('unicode',str)]:
        if k not in np.__dict__:setattr(np,k,v)
    if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
    import chumpy
    with SOURCE.open('rb') as f:data=pickle.load(f,encoding='latin1')
    converted=arrays(data)
    assert converted['v_template'].shape==(6890,3)
    for k in data:
        if type(data[k]).__module__.startswith('chumpy.'):
            np.testing.assert_array_equal(data[k].r,converted[k])
    before=smplx.SMPL(str(SOURCE),gender='neutral',num_betas=10)
    expected,checks=cases(before)
    DEST.mkdir()
    target=DEST/'SMPL_NEUTRAL.pkl'
    with target.open('xb') as f:pickle.dump(converted,f,protocol=2)
    after=smplx.SMPL(str(target),gender='neutral',num_betas=10);actual,_=cases(after)
    for k in expected:np.testing.assert_allclose(expected[k],actual[k],atol=1e-7,rtol=1e-6)
    np.savez(DEST/'cpu_reference.npz',**expected)
    record=dict(status='numpy_container_conversion_equivalence_passed',source=str(SOURCE),source_sha256=DIGEST,
        output=str(target),output_sha256=sha(target),shapedirs_shape=list(converted['shapedirs'].shape),
        converted_keys=[k for k in data if type(data[k]).__module__.startswith('chumpy.')],
        no_shape_basis_truncation=True,checks=checks,torch_version=torch.__version__,numpy_version=np.__version__,
        smplx_module=smplx.__file__,script_sha256=sha(__file__),jacobian_runtime_validated=False,
        provenance='Community mirror; native official SMPLify file equivalence not independently established')
    (P/'state/jacobian_smpl_neutral_conversion_v1.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

def validate():
    rec=json.loads((P/'state/jacobian_smpl_neutral_conversion_v1.json').read_text());target=Path(rec['output']);assert sha(target)==rec['output_sha256']
    model=smplx.create(str(target),model_type='smpl',gender='neutral',num_betas=10)
    result,checks=cases(model);errors={}
    with np.load(DEST/'cpu_reference.npz',allow_pickle=False) as baseline:
        for k,v in result.items():
            errors[k]=float(np.abs(v-baseline[k]).max());np.testing.assert_allclose(v,baseline[k],atol=2e-6,rtol=2e-5)
    link=P/'sources/jacobianavatar/resource/smpl/SMPL_NEUTRAL.pkl';link.parent.mkdir(exist_ok=True,parents=True)
    assert not link.exists() and not link.is_symlink();link.symlink_to(target)
    record=dict(status='jacobian_smpl_cpu_forward_backward_passed_and_linked',source_conversion=str(P/'state/jacobian_smpl_neutral_conversion_v1.json'),
        output=str(target),output_sha256=sha(target),consumer_link=str(link),checks=checks,reference_max_errors=errors,
        torch_version=torch.__version__,numpy_version=np.__version__,smplx_module=smplx.__file__,script_sha256=sha(__file__),
        full_jacobian_ready=False,scope='SMPL neutral component only; transfer mapping, MHR runtime and complete fitting/training remain')
    (P/'state/jacobian_smpl_neutral_validation_v1.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');a=p.parse_args()
    validate() if a.validate else convert()
