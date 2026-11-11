"""Validate the bounded public tree without reading credentials or private data."""
import ast,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','artifacts','private','third_party','__pycache__','.pytest_cache'}
errors=[];count=0
for p in ROOT.rglob('*'):
    if not p.is_file() or any(x in SKIP or x.endswith('.egg-info') for x in p.relative_to(ROOT).parts):continue
    count+=1
    if p.is_symlink():errors.append(f'Symlink not allowed in public tree: {p.relative_to(ROOT)}');continue
    if p.stat().st_size>2*1024**2:errors.append(f'Oversized source file: {p.relative_to(ROOT)}')
    if p.suffix in {'.npz','.npy','.pt','.pth','.ckpt','.pkl','.safetensors','.ply','.mp4','.jpg','.png','.pem'}:
        errors.append(f'Unexpected binary/data file: {p.relative_to(ROOT)}');continue
    try:s=p.read_text()
    except UnicodeDecodeError:errors.append(f'Unexpected binary: {p.relative_to(ROOT)}');continue
    if re.search(r'(?:/mnt|/home)/'+r'(?:ojc|wmj)(?:/|\b)',s):errors.append(f'Personal server path: {p.relative_to(ROOT)}')
    if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',s):errors.append(f'Credential pattern: {p.relative_to(ROOT)}')
    if p.suffix=='.py':
        try:ast.parse(s)
        except SyntaxError as e:errors.append(f'Python syntax: {p.relative_to(ROOT)}: {e}')
    if p.suffix=='.json':
        try:json.loads(s)
        except ValueError as e:errors.append(f'JSON: {p.relative_to(ROOT)}: {e}')
    if p.suffix=='.md':
        for link in re.findall(r'\]\(([^)]+)\)',s):
            if '://' in link or link.startswith('#'):continue
            if not (p.parent/link.split('#')[0]).exists():errors.append(f'Broken doc link: {p.relative_to(ROOT)} -> {link}')
assert not errors,'\n'.join(errors)
print(f'Public tree validated: {count} text files; no oversized binaries, personal server paths, or credential patterns.')

