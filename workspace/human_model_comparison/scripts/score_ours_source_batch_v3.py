"""Score newly certified Ours pairs with the unchanged frozen core implementation."""
import csv,fcntl,json,time
from pathlib import Path
import torch
from PIL import Image
import human_metric_batch_v1 as b

H=b.H
def main():
    out=H/'reports/source_camera_metric_cohort_v3';sp=out/'summary.json'
    with (out/'scorer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        summary=b.read(sp);b.verified(summary['selection'],summary['selection_sha256'])
        active=b.read(H/'inputs/active_human_benchmark.json');assert active['selection_sha256']==summary['selection_sha256']
        selected=b.read(summary['selection'])['objects'];ids=[r['id'] for r in selected]
        assert len(ids)==100 and 'HN0035' not in ids
        b.verified(summary['protocol'],summary['protocol_sha256']);protocol=b.read(summary['protocol'])
        b.verified(b.m.__file__,protocol['code_sha256'])
        history=out/'history';history.mkdir(exist_ok=True);backup=history/(b.m.sha(sp)+'.json')
        if not backup.exists():backup.write_bytes(sp.read_bytes())
        for p,digest in summary['cached_result_sha256'].items():b.verified(p,digest)
        certpath=H/'reports/ours_source_batch_v3/manifest.json';certs=b.read(certpath)
        assert certs['selection_sha256']==summary['selection_sha256']
        cfg=protocol['lpips'];torch.set_num_threads(4)
        metric=b.m.AlexLPIPS(cfg['backbone'],cfg['linear'],cfg['backbone_sha256'],cfg['linear_sha256'],'cpu')
        done=[];skipped=[];newly_computed=[];reused=[]
        for entry in certs['rows']:
            row=next(r for r in summary['rows'] if r['id']==entry['id'] and r['method']=='ours')
            ref=row['reference']
            if ref is None or not ref['reference_eligible']:
                skipped.append(dict(id=entry['id'],reason='reference_qualification_pending_or_ineligible'));continue
            prepared=dict(id=entry['id'],method='ours',reference=ref,registration_record=entry['registration_record'],registration_sha256=entry['registration_sha256'])
            prepared['input_key']=b.digest(dict(reference=ref,registration_sha256=entry['registration_sha256'],protocol_sha256=summary['protocol_sha256'],metric_code_sha256=b.m.sha(b.m.__file__),runner_sha256=b.m.sha(b.__file__)))
            b.check_registration(ref,'ours',prepared)
            path=b.result_path(out,prepared)
            if path.exists():
                result=b.read(path);assert result['input_key']==prepared['input_key'] and result['status']=='scored';reused.append(entry['id'])
            else:
                result,a,pred=b.score_pair(prepared,metric);media=out/'media'/entry['id'];media.mkdir(parents=True,exist_ok=True)
                for name,array in [('reference',a['rgb']),('ours',pred['rgb']),('ours_alpha',pred['mask'])]:Image.fromarray(array).save(media/(name+'.png'))
                result.update(lpips=metric.provenance,active_selection_sha256=summary['selection_sha256'],execution_script_sha256=b.m.sha(__file__))
                b.save(path,result);newly_computed.append(entry['id'])
            row.update(status='scored',reason=None,result=str(path),result_sha256=b.m.sha(path),input_key=prepared['input_key'],metrics=result['metrics'],reused_from_previous_cohort=False)
            done.append(entry['id']);print(entry['id'],'scored',flush=True)
        methods=list(summary['coverage']);by_method={m:{} for m in methods}
        for row in summary['rows']:
            if row['status']=='scored':
                b.verified(row['result'],row['result_sha256']);result=b.read(row['result']);assert result['metrics']==row['metrics'];by_method[row['method']][row['id']]=result
        names=['PSNR','SSIM','LPIPS','IoU','IoU_raw'];pairs={}
        for chosen in [methods,['omnire','splatad'],['ours','omnire','splatad']]:
            pairs['+'.join(chosen)]={n:b.m.paired_summary({m:by_method[m] for m in chosen},n,ids) for n in names}
        summary.update(status='active_cohort_core_metrics_incremental',coverage={m:len(v) for m,v in by_method.items()},pairings=pairs,
                       additional_ours_scores=done,reference_gated_skips=skipped,cached_scores_recomputed=0,newly_computed_in_this_run=newly_computed,reused_ours_in_this_run=reused,additional_certification=str(certpath),additional_certification_sha256=b.m.sha(certpath),
                       updated_at=time.time(),additional_scoring_script_sha256=b.m.sha(__file__))
        b.save(sp,summary)
        fields=['id','method','status','reason','result','result_sha256']+[k for n in names+['ED-P'] for k in [n,n+'_status']]
        with (out/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in summary['rows']:
                item={k:r.get(k) for k in fields[:6]}
                for n in names+['ED-P']:
                    v=r.get('metrics',{}).get(n,dict(status='na',value=None));item[n+'_status']=v['status'];item[n]='inf' if v['status']=='positive_infinity' else v['value'] if v['status']=='ok' else 'NA'
                writer.writerow(item)
        print(json.dumps(dict(coverage=summary['coverage'],added=done,skipped=skipped)))

if __name__=='__main__':main()
