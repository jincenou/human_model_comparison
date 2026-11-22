"""Immutable cohort/reference contract and live 300-row export inventory."""
import collections
import csv
import io
import json
from pathlib import Path
import time

from score_streetgs_human_static_v1 import read, sha, checked, save, references

H = Path(__file__).resolve().parents[1]
R = H.parent / 'vehicle_model_comparison'
ROOT = H / 'reports/native_human_export_reaudit_v1'
EXPORTS = H / 'exports/native_human_export_reaudit_v1'
METHODS = ('streetgs', 'splatad', 'omnire')
SELECTION_SHA = '03110e1d403248ff84d4f108a23602aad153781c76530fbc572bd5bc22324d55'


def freeze():
    active, source, refs = references()
    assert active['selection_sha256'] == SELECTION_SHA
    inherited = {
        'appearance': R / 'reports/source_appearance_alignment_20261115_v1/protocol.json',
        'geometry': H / 'evaluation_static_cd_fscore_time_v1/protocol.json',
        'geometry_references': H / 'evaluation_static_cd_fscore_time_v1/reference_manifest.json',
        'edp': H / read(H / 'inputs/active_results.json')['edp_root'] / 'protocol.json',
    }
    for ref in refs.values():
        for key in ('camera_contract', 'reference_rgb', 'reference_mask'):
            checked(ref[key], ref[key + '_sha256'])
    contract = dict(schema='native_human_export_reaudit_v1', selection=active['selection'],
        selection_sha256=SELECTION_SHA, reference_summary=str(source), reference_summary_sha256=sha(source),
        methods=list(METHODS), expected_people=100, expected_method_records=300,
        ids=sorted(refs), references=refs,
        inherited_protocols={k:dict(path=str(p),sha256=sha(p)) for k,p in inherited.items()},
        policy=dict(selection='Frozen reference mask; native foreground visible contribution > 0; retain original target actor where present',
                    quality='No score-based output selection, deletion or threshold search; quality failures remain documented',
                    exposure='Unchanged physical source exposure and camera; neighbor views diagnostic only',
                    image='Independent white-composited native RGB and own alpha; reference mask never erases prediction pixels',
                    geometry='Unchanged observed-surface reference panel and scale/depth rules',
                    temporal='Static background support is not native learned human motion; no invented trajectories',
                    aggregation='Version A: method-own metric-valid identities, no forced intersection or zero imputation',
                    disk_reserve_gib=10, historical_outputs_modified=False))
    p = ROOT / 'batch_contract.json'
    if p.exists():
        assert read(p) == contract, 'Frozen contract changed; do not silently rebind this batch'
    else:
        save(p, contract)
    return contract


def validations(method):
    records = {}
    for p in sorted((EXPORTS / method).rglob('validation.json')):
        row = read(p); sid = row.get('id', row.get('sample_id'))
        if not sid:
            continue
        assert sid not in records, ('Duplicate final validation', method, sid, str(p), records[sid][0])
        records[sid] = (p, row)
    return records


def inventory():
    contract = freeze()
    rows=[]
    for method in METHODS:
        found=validations(method)
        assert set(found) <= set(contract['ids'])
        review_path = ROOT / method / 'semantic_review.json'
        review = {x['id']: x for x in read(review_path).get('rows', [])} if review_path.exists() else {}
        edp = {}
        for ep in sorted((ROOT / 'metrics' / method / 'edp').glob('*/results.json')):
            for item in read(ep)['rows']:
                assert item['id'] not in edp, ('Duplicate ED-P result', method, item['id'])
                edp[item['id']] = item
        for sid in contract['ids']:
            ref=contract['references'][sid]
            p,v=found.get(sid,(None,{}))
            row=dict(id=sid,method=method,identity_key=ref['identity_key'],timestamp_us=ref['timestamp_us'],
                export_status=v.get('status','not_started'),validation=str(p) if p else None,
                validation_sha256=sha(p) if p else None,reload_verified=v.get('reload_verified',False),
                source_support_status=v.get('source_support_status'),reason=v.get('reason'),
                actor_node_present=v.get('actor_node_present'),source_reference_mask_sha256=ref['reference_mask_sha256'])
            score=ROOT/'metrics'/method/'static_rows'/(sid+'.json')
            if score.exists():
                s=read(score);row.update(static_status=s['status'],static_result=str(score),static_result_sha256=sha(score))
            else:row.update(static_status='not_scored',static_result=None,static_result_sha256=None)
            geom=ROOT/'geometry'/method/'results'/(sid+'.json')
            row['geometry_status']=read(geom)['status'] if geom.exists() else 'not_scored'
            inspected = review.get(sid, {})
            row.update(semantic_review_label=inspected.get('label'),
                       semantic_review_note=inspected.get('note'),
                       semantic_review_source=str(review_path) if inspected else None,
                       neighbor_check_required=inspected.get('neighbor_check_required'),
                       edp_status=edp.get(sid, {}).get('score', {}).get('status', 'not_scored'),
                       edp_na_reason=edp.get(sid, {}).get('score', {}).get('reason'),
                       edp_valid_part_count=edp.get(sid, {}).get('valid_part_count'))
            rows.append(row)
    assert len(rows)==300 and len({(r['id'],r['method']) for r in rows})==300
    totals={m:dict(export=dict(collections.Counter(r['export_status'] for r in rows if r['method']==m)),
                   static=dict(collections.Counter(r['static_status'] for r in rows if r['method']==m)),
                   geometry=dict(collections.Counter(r['geometry_status'] for r in rows if r['method']==m)),
                   edp=dict(collections.Counter(r['edp_status'] for r in rows if r['method']==m)),
                   semantic=dict(collections.Counter(r['semantic_review_label'] or 'pending' for r in rows if r['method']==m))) for m in METHODS}
    save(ROOT/'inventory.json',dict(contract_sha256=sha(ROOT/'batch_contract.json'),updated_at=time.time(),rows=rows,totals=totals))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'inventory.csv').write_text(buf.getvalue())
    print(json.dumps(totals,ensure_ascii=False))
    return rows


if __name__=='__main__':inventory()
