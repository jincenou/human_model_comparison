"""Enter the unchanged native Grid4D environment for independent reload."""
import argparse,os,subprocess,types
from pathlib import Path
E=Path(__file__).resolve().parents[1];CODE=E.parent.parent/'human';PROD=E.parent.parent/'asset_work/human_nuscenes100_20261026'
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    mod=types.ModuleType('native_environment');mod.__file__=str(CODE/'common.py');exec(compile((CODE/'common.py').read_text(),mod.__file__,'exec'),mod.__dict__)
    env=mod.environment(os.environ['CUDA_VISIBLE_DEVICES'],'grid4d');env.update(HUMAN_RUN_ROOT=a.run,HUMAN_ASSET_ROOT=str(Path(a.run)/'native_assets'))
    for k,sub in [('GRID4D_HASH_BUILD','hash'),('TORCH_EXTENSIONS_DIR','extensions'),('CUDA_CACHE_PATH','cuda')]:env[k]=str(PROD/'cache'/sub)
    raise SystemExit(subprocess.call([str(mod.PYTHONS['grid4d']),'-B','-u',str(E/'scripts/verify_ours_time.py'),'--run',a.run],env=env,cwd=CODE))
if __name__=='__main__':main()
