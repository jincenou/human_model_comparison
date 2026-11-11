"""Incrementally scan scene assets and preserve immutable certification batches.

Reuses the inspected geometry/time checks from the v1 auditor. It never trains,
re-renders, fits a pose, or promotes missing structured evidence to a pass.
"""
import json,hashlib,traceback
from pathlib import Path
from collections import Counter
import audit_scene_source_registration_v1 as a
H=a.H
ROOT=H/'reports/scene_source_registration_incremental_v2'
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 qpath=a.C/'reference_qualification_frozen_v1.json';q=a.read(qpath);draft=[];certificates=[]
 for ref in q['objects']:
  for method in a.METHODS:
   row=dict(id=ref['id'],method=method,reference_eligible=ref['reference_eligible'],metric_eligible=False)
   if not ref['reference_eligible']:row.update(status='na',reason='reference_ineligible',detail=ref['na_reasons'])
   elif method not in ['omnire','splatad']:row.update(status='na',reason='source_registration_not_yet_validated')
   else:
    p,script=a.source_for(method,ref['id'])
    if not p.exists():row.update(status='na',reason='native_source_render_missing')
    else:
     try:
      v=a.read(p)
      if v.get('status')!='passed':raise ValueError('Native render unavailable: '+str(v.get('reason',v.get('status'))))
      if a.m.sha(script)!=v.get('extraction_script_sha256'):raise ValueError('Renderer source hash differs from historical pilot; revalidate from archived source or rerender')
      certificate=a.certify(ref,method,p,script);certificates.append(certificate);row.update(status='certified_for_core_metrics',metric_eligible=True)
     except (AssertionError,KeyError,ValueError,FileNotFoundError) as e:
      row.update(status='na',reason='registration_evidence_incomplete_or_failed',detail=str(e) or 'Structured geometry/time assertion failed',failed_check=traceback.extract_tb(e.__traceback__)[-1].line,render_validation=str(p),render_validation_sha256=a.m.sha(p))
   draft.append(row)
 evidence=dict(qualification_sha256=a.m.sha(qpath),auditor_sha256=a.m.sha(a.__file__),scanner_sha256=a.m.sha(__file__),rows=draft,certificates=certificates)
 key=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest();out=ROOT/'batches'/key
 if (out/'manifest.json').exists():
  manifest=a.read(out/'manifest.json')
  for row in manifest['rows']:
   if row['metric_eligible']:assert a.m.sha(row['registration_record'])==row['registration_sha256']
 else:
  rows=[dict(x) for x in draft]
  for cert in certificates:
   p=out/cert['identity']/(cert['method']+'.json');save(p,cert)
   row=next(x for x in rows if x['id']==cert['identity'] and x['method']==cert['method']);row.update(registration_record=str(p),registration_sha256=a.m.sha(p))
  manifest=dict(status='incremental_scene_source_registration_audited',batch_key=key,qualification=str(qpath),qualification_sha256=a.m.sha(qpath),
   rows=rows,summary={method:dict(Counter(r['status'] for r in rows if r['method']==method)) for method in a.METHODS},
   auditor_sha256=a.m.sha(a.__file__),scanner_sha256=a.m.sha(__file__),archive=str(out/'manifest.json'),selected_denominator=100,
   source_reconstruction_only=True,full_five_method_comparison_complete=False)
  save(out/'manifest.json',manifest)
 save(ROOT/'latest_manifest.json',manifest);print(json.dumps(dict(manifest=str(ROOT/'latest_manifest.json'),summary=manifest['summary'])))
if __name__=='__main__':main()
