"""Native activated TRELLIS arrays, existing gsplat environment, metrology only."""
import sys
from common import *
from depth import GaussianDepth
from pilot import measure
out=Path(sys.argv[1]);job=read(out/'decoded_job.json');adapter=GaussianDepth(job)
if '--compare' in sys.argv:
    with np.load(out/'native_perspective_check.npz') as z:
        ed,alpha=adapter.render(z['V'],z['K'],512,.01,1000,force_gsplat=True)
    np.savez(out/'gsplat_perspective_check.npz',ed=ed,alpha=alpha)
else:measure(adapter,np.asarray(read(out/'axes.json')),out)
