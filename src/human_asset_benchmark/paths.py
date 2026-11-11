"""Portable roots for the released workspace adapters and metadata templates."""
import argparse
import json
import os
import re
from pathlib import Path

def roots():
    repo = Path(__file__).resolve().parents[2]
    home = Path.home()
    workspace = os.environ.get('HMC_WORKSPACE', str(repo/'workspace'))
    defaults = {
        'HMC_WORKSPACE': workspace,
        'HMC_ENV_ROOT': str(home/'conda_envs'),
        'HMC_WEIGHT_ROOT': str(home/'model_weights/human_comparison'),
        'HMC_MODEL_CACHE': str(home/'model_weights'),
        'HMC_SYSTEM_ROOT': str(home),
        'HMC_STORAGE_ROOT': str(repo/'private'),
        'HMC_THIRD_PARTY': str(repo/'third_party'),
        'CONDA_ROOT': str(home/'miniconda3'),
        'NUSCENES_ROOT': str(repo/'private/nuScenes'),
        'SPLATAD_ROOT': str(repo/'third_party/splatad'),
        'DRIVESTUDIO_ROOT': str(repo/'third_party/drivestudio'),
    }
    return {k: os.environ.get(k,v) for k,v in defaults.items()}

def expand(value):
    mapping=roots()
    def replace(match):
        key=match.group(1)
        if key not in mapping:raise ValueError('Unknown path root: '+key)
        return mapping[key]
    return re.sub(r'\$\{([A-Z_]+)\}',replace,value)

def materialize(value):
    if isinstance(value,str):return expand(value)
    if isinstance(value,list):return [materialize(v) for v in value]
    if isinstance(value,dict):return {expand(k):materialize(v) for k,v in value.items()}
    return value

def main():
    p=argparse.ArgumentParser(description='Expand a metadata template without overwriting an existing file.')
    p.add_argument('template',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    data=materialize(json.loads(a.template.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(data,f,indent=2);f.write('\n')


if __name__ == "__main__":
    main()
