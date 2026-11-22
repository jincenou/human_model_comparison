"""Paths, content bindings and atomic writes for the new human experiment only."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1]
H = E.parent
R = H.parent / 'vehicle_model_comparison'
D = H / 'handoff/human_static_cd_fscore_time_execution_20261114_v2'
G = R / 'evaluation_geometry_expanded_20261114_v2'
METHODS = ['ours', 'ah', 'trellis_original', 'trellis2', 'omnire', 'splatad']
SELECTION_SHA = '03110e1d403248ff84d4f108a23602aad153781c76530fbc572bd5bc22324d55'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def checked(p, digest):
    assert sha(p) == digest, ('Changed input', str(p))

def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    t.replace(p)

def immutable(p, d):
    p = Path(p)
    if p.exists():
        assert read(p) == d, ('Immutable record differs', str(p))
    else:
        save(p, d)

def selection():
    a = read(H/'inputs/active_human_benchmark.json')
    assert a['selection_sha256'] == SELECTION_SHA
    checked(a['selection'], SELECTION_SHA)
    d = read(a['selection']); ids = {x['id']: x for x in d['objects']}
    assert len(ids) == 100 and 'HN0646' in ids and 'HN0035' not in ids
    return a, ids

def csv_write(p, rows):
    import csv
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def rigid_part(T):
    T = np.array(T, dtype=float)
    s = np.cbrt(np.linalg.det(T[:3,:3])); assert s > 0
    V = T.copy(); V[:3] /= s
    np.testing.assert_allclose(V[:3,:3] @ V[:3,:3].T, np.eye(3), atol=2e-6)
    return V, float(s)
