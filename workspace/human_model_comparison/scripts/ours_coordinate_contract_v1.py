"""Analytic rigid-frame contract, independent of raster alpha cutoff decisions."""
import numpy as np
from scipy.spatial.transform import Rotation

def covariance(cloud):
    q=cloud.quaternions.astype(np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    axes=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    return (axes*cloud.scales[:,None,:].astype(np.float64)**2)@axes.transpose(0,2,1)

def evaluate(original,camera,transformed,new_camera):
    def camera_values(cloud,cam):
        xyz=(cloud.means.astype(np.float64)-cam[:3,3])@cam[:3,:3]
        cov=cam[:3,:3].T@covariance(cloud)@cam[:3,:3]
        direction=cloud.means.astype(np.float64)-cam[:3,3]
        direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        return xyz,cov,direction@cloud.local_from_appearance
    xyz,cov,direction=camera_values(original,camera)
    new_xyz,new_cov,new_direction=camera_values(transformed,new_camera)
    result=dict(camera_space_position_max_error=float(np.abs(xyz-new_xyz).max()),
        camera_space_covariance_relative_frobenius_max=float((np.linalg.norm(cov-new_cov,axis=(1,2))/np.maximum(np.linalg.norm(cov,axis=(1,2)),1e-20)).max()),
        appearance_direction_max_error=float(np.abs(direction-new_direction).max()),
        sh_exact_equal=bool(np.array_equal(original.sh,transformed.sh)),
        opacity_exact_equal=bool(np.array_equal(original.opacities,transformed.opacities)))
    result['passed']=(result['camera_space_position_max_error']<1e-6 and
        result['camera_space_covariance_relative_frobenius_max']<2e-6 and
        result['appearance_direction_max_error']<1e-6 and result['sh_exact_equal'] and result['opacity_exact_equal'])
    return result

def raster_difference(a,b):
    diff=np.abs(a-b)
    return dict(max=float(diff.max()),mean=float(diff.mean()),rms=float(np.sqrt(np.mean(diff**2))),
        count_gt_002=int((diff>.002).sum()),count_gt_one_byte=int((diff>1/255+1e-5).sum()),elements=int(diff.size))

def validate_raster(stats):
    # This bound does not replace the stronger analytic checks above. A float32
    # perturbation can cross gsplat's ALPHA_THRESHOLD=1/255 at isolated pixels.
    return stats['max']<=1/255+1e-5 and stats['mean']<1e-6 and stats['rms']<2e-5
