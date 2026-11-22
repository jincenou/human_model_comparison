"""Bounded semantic repair for mirrored textures on both faces of thin assets."""
import recover_registration as rr
from common import *
def main():
    out=[]
    for job in read(E/'intake.json')['rows']:
        if job['method'] not in ['trellis_original','trellis2'] or not job['asset_available'] or job.get('T_source_camera_from_native'):continue
        for center in [0,4,8,12]:
            indices={(center+i)%16 for i in [-2,-1,0,1,2]}
            records=[read(r['record']) for r in read(rr.root(job)/'keypoints.json')['records']]
            records=[r for r in records if r['case']['role']=='reference' or r['case']['view_index'] in indices]
            result=rr.register(job,records,E/'registration_recovery/local_views'/job['method']/job['id']/str(center));result.update(center_view=center,local_view_script_sha256=sha(__file__));out.append(result)
    save(E/'registration_recovery/local_views_summary.json',dict(rows=out,reason='Opposite sheet faces repeat mirrored appearance; all-view pose-label consensus collapsed paired joints. Fixed local angular neighborhoods avoid mixing opposite-face semantics. All54 gaps evaluated; no score used.'))
if __name__=='__main__':main()
