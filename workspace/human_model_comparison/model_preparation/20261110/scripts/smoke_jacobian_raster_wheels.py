"""Forward/backward checks from built wheels, without installing into the env."""
from human_asset_benchmark.paths import expand as _hmc_expand
from pathlib import Path
import hashlib,json,sys,zipfile,os
P=Path(__file__).resolve().parents[1]
ROOT=Path(_hmc_expand('${HMC_SYSTEM_ROOT}/human_model_tmp/jacobian_cuda_sources'))
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(Path(os.environ['CUDA_HOME'])/'bin')+os.pathsep+os.environ.get('PATH','')
cuda=Path(os.environ['CUDA_HOME'])
os.environ['CPATH']=os.pathsep.join(str(x)for x in [cuda/'targets/x86_64-linux/include',cuda/'include'])
os.environ['LIBRARY_PATH']=os.pathsep.join(str(x)for x in [cuda/'targets/x86_64-linux/lib',cuda/'lib'])
state=json.loads((P/'state/jacobian_cuda_wheels_20261113_v2.json').read_text())
assert state['status']=='wheels_built_install_and_runtime_pending'
target=ROOT/'wheel_runtime';target.mkdir(exist_ok=True)
for name,digest in state['wheels'].items():
    path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    with zipfile.ZipFile(path)as z:
        for member in z.namelist():assert target.resolve()in(target/member).resolve().parents
        z.extractall(target)
sys.path.insert(0,str(target))
import torch
assert torch.__version__=='2.5.1+cu121' and torch.cuda.is_available()
from diff_gauss import GaussianRasterizer,GaussianRasterizationSettings
def tensor(a,grad=False):return torch.tensor(a,dtype=torch.float32,device='cuda',requires_grad=grad)
settings=GaussianRasterizationSettings(image_height=32,image_width=32,tanfovx=1.,tanfovy=1.,
    bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=torch.eye(4,device='cuda'),
    projmatrix=tensor([[1,0,0,0],[0,1,0,0],[0,0,1.001,1],[0,0,-.1001,0]]),
    sh_degree=0,campos=torch.zeros(3,device='cuda'),prefiltered=False,debug=False)
position=tensor([[0,0,2]],True);xy=tensor([[0,0,0]],True)
rgb=tensor([[.8,.2,.1]],True);opacity=tensor([[.8]],True)
scale=tensor([[.1,.1,.1]],True);rotation=tensor([[1,0,0,0]],True)
out=GaussianRasterizer(settings)(means3D=position,means2D=xy,opacities=opacity,
    colors_precomp=rgb,scales=scale,rotations=rotation)
assert len(out)==6 and all(torch.isfinite(x).all()for x in out)
assert int((out[3]>.01).sum())>0
out[0].sum().backward();assert rgb.grad is not None and float(rgb.grad.abs().sum())>0
assert all(t.grad is not None and torch.isfinite(t.grad).all()for t in [position,xy,rgb,opacity,scale,rotation])
result=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),environment_installed=False,
    wheel_sha256=state['wheels'],diff_gauss=dict(status='forward_backward_passed',alpha_max=float(out[3].max()),
    rgb_gradient_l1=float(rgb.grad.abs().sum())),full_method_validated=False)
save=P/'state/jacobian_raster_runtime_20261113.json';save.write_text(json.dumps(result,indent=2))
print('diff_gauss forward/backward passed',flush=True)
import nvdiffrast.torch as dr
context=dr.RasterizeCudaContext()
vertices=tensor([[[-.8,-.8,0,1],[.8,-.8,0,1],[0,.8,0,1]]],True)
triangles=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
rast,_=dr.rasterize(context,vertices,triangles,resolution=[32,32])
assert torch.isfinite(rast).all() and int((rast[...,3]>0).sum())>0
rast[...,:2].square().sum().backward();assert torch.isfinite(vertices.grad).all() and float(vertices.grad.abs().sum())>0
result['nvdiffrast']=dict(status='cuda_forward_backward_passed',covered_pixels=int((rast[...,3]>0).sum()),vertex_gradient_l1=float(vertices.grad.abs().sum()))
save.write_text(json.dumps(result,indent=2));print('nvdiffrast CUDA forward/backward passed',flush=True)
