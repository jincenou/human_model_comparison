"""Compare compiled GPU neighbor operations with brute-force torch distances."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import argparse, hashlib, json, sys, zipfile

P=Path(__file__).resolve().parents[1]
ROOT=Path(_hmc_expand('${HMC_STORAGE_ROOT}/human_model_tmp/jacobian_native_wheels'))
parser=argparse.ArgumentParser();parser.add_argument('module',choices=['simple_knn','pytorch3d']);args=parser.parse_args()
state=json.loads((P/'state/jacobian_remaining_wheels_20261113.json').read_text())
assert any(s['name']==args.module and s['status']=='built'for s in state['stages'])
wheels=list((ROOT/'wheels').glob(args.module+'-*.whl'));assert len(wheels)==1
wheel=wheels[0];target=ROOT/'runtime'/args.module;target.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(wheel)as z:
    for name in z.namelist():assert target.resolve()in(target/name).resolve().parents
    z.extractall(target)
sys.path.insert(0,str(target))
import torch
torch.manual_seed(7)
assert torch.__version__=='2.5.1+cu121' and torch.cuda.is_available()
points=torch.randn(64,3,device='cuda')
distances=(points[:,None]-points[None,:]).square().sum(-1)
if args.module=='simple_knn':
    from simple_knn._C import distCUDA2
    observed=distCUDA2(points)
    distances.fill_diagonal_(float('inf'))
    expected=distances.topk(3,largest=False).values.mean(-1)
    torch.testing.assert_close(observed,expected,rtol=1e-5,atol=1e-6)
    checks=['GPU mean squared distance to 3 nearest other points matches exhaustive search']
else:
    from pytorch3d.ops import knn_points
    points.requires_grad_()
    result=knn_points(points[None],points[None],K=4)
    expected=distances.topk(4,largest=False).values
    observed=result.dists[0]
    torch.testing.assert_close(observed,expected,rtol=1e-5,atol=1e-6)
    observed.sum().backward();assert torch.isfinite(points.grad).all() and points.grad.abs().sum()>0
    checks=['GPU KNN matches exhaustive search','finite nonzero position gradients']
record=dict(status='component_passed',module=args.module,checks=checks,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            wheel=str(wheel),sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),max_error=float((observed-expected).abs().max()),
            environment_installed=False,full_method_validated=False)
(P/'state'/('jacobian_'+args.module+'_runtime_20261113.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
