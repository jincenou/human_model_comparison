from human_asset_benchmark.paths import expand as _hmc_expand
import argparse,os,subprocess,sys
import numpy as np
from PIL import Image
from common import *
from scene_depth import prepare,SceneDepth
from pilot import measure

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--id',required=True);a=p.parse_args()
    job=next(j for j in read(E/'intake.json')['rows'] if j['id']==a.id and j['method']==a.method)
    out=E/'pilot'/a.method/a.id;pack,rec=prepare(job,out);adapter=SceneDepth(pack)
    camera=read(job['camera_contract']);Tworld=np.asarray(pack['T_world_from_native']);axes=rigid_part(Tworld)[0][:3,:3]
    # Source alpha equivalence to actual native renderer, no metric used.
    V=np.linalg.inv(np.asarray(camera['T_world_from_camera_opencv']))@Tworld
    ed,alpha=adapter.render(V,np.asarray(camera['K_raw']),camera['render_wh'][0],.01,1000.)
    cert=read(job['source_certificate']);original=np.asarray(Image.open(cert['source_alpha']).convert('L'))
    diff=abs((alpha[:original.shape[0]]*255).round().clip(0,255)-original.astype(float))
    checks=dict(source_alpha_max=float(diff.max()),source_alpha_mean=float(diff.mean()),shape=list(original.shape),prepared_sha256=rec['prepared_sha256'])
    if a.method=='splatad':
        ev=dict(os.environ,TORCH_EXTENSIONS_DIR=str(H/'cache/torch_ext/ours'),CUDA_HOME=_hmc_expand('${CONDA_ROOT}/envs/ab_eval'));ev['PATH']=_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin:')+ev['PATH']
        subprocess.run([_hmc_expand('${CONDA_ROOT}/envs/ab_eval/bin/python'),'-B',str(E/'scripts/measure_splatad.py'),str(out)],env=ev,check=True)
        data=read(out/'measurement.json')
    else:data=measure(adapter,axes,out)
    save(out/'adapter_checks.json',checks);print(a.method,a.id,data['status'],checks,flush=True)

if __name__=='__main__':main()
