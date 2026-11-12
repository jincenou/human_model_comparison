"""Six-part cosine distance on the frozen common ROI; missing parts stay NA.

Skeleton definitions follow Asset Harvester's Apache-2.0 embedding_metrics.py
(NVIDIA, 2026). This common-ROI variant does not reproduce its independent crops,
four-part cutoff, SAM3D detector or ViT-H+ encoder. See the recorded protocol.
"""
import numpy as np
from human_source_metrics_v1 import PARTS, aggregate_visible_parts

SEGMENTS = (((0,1),(0,2),(1,3),(2,4)), ((5,6),(5,11),(6,12),(11,12)),
            ((5,7),(7,9)), ((6,8),(8,10)), ((11,13),(13,15)), ((12,14),(14,16)))


def part_labels(mask, keypoints, patch_size=16, confidence=.3, max_distance_patches=2, excluded_parts=()):
    """Assign foreground patch centres to confidently supported skeleton parts."""
    mask=np.asarray(mask);h,w=mask.shape
    assert h%patch_size==w%patch_size==0
    y,x=np.meshgrid((np.arange(h//patch_size)+.5)*patch_size,
                    (np.arange(w//patch_size)+.5)*patch_size,indexing='ij')
    points=np.stack([x.ravel(),y.ravel()],axis=1)
    distances=np.full((len(PARTS),len(points)),np.inf)
    if keypoints is not None:
        k=np.asarray(keypoints,np.float64);assert k.shape==(17,3) and np.isfinite(k).all()
        usable=(k[:,2]>=confidence)&(k[:,0]>=0)&(k[:,0]<w)&(k[:,1]>=0)&(k[:,1]<h)
        for part,segments in enumerate(SEGMENTS):
            for a,b in segments:
                if not (usable[a] and usable[b]):continue
                delta=k[b,:2]-k[a,:2];length=float(delta@delta)
                t=np.clip((points-k[a,:2])@delta/max(length,1e-20),0,1)
                d=np.linalg.norm(points-(k[a,:2]+t[:,None]*delta),axis=1)
                distances[part]=np.minimum(distances[part],d)
    best=distances.argmin(axis=0)
    valid=(distances.min(axis=0)<=max_distance_patches*patch_size)
    valid &= mask[points[:,1].astype(int),points[:,0].astype(int)]>127
    labels=np.where(valid,best,-1).astype(np.int8).reshape(h//patch_size,w//patch_size)
    # Exclude after assignment: a hidden head's patches must not be reassigned
    # to the torso merely because that head is excluded by reference evidence.
    for name in excluded_parts:
        labels[labels==PARTS.index(name)]=-1
    return labels


def pool_parts(features, labels):
    """Mean final-layer patch tokens; no reference mask is used on predictions."""
    features=np.asarray(features,np.float32)
    assert features.ndim==3 and features.shape[:2]==labels.shape and np.isfinite(features).all()
    embeddings=np.zeros((6,features.shape[2]),np.float64)
    counts=np.array([np.count_nonzero(labels==i) for i in range(6)],dtype=np.int64)
    for i,n in enumerate(counts):
        if n:embeddings[i]=features[labels==i].astype(np.float64).mean(axis=0)
    return embeddings,counts


def compare_parts(reference, prediction, reference_counts, prediction_counts):
    reference=np.asarray(reference,np.float64);prediction=np.asarray(prediction,np.float64)
    assert reference.shape==prediction.shape and reference.shape[0]==6
    values={};details={}
    for i,name in enumerate(PARTS):
        nr,npred=int(reference_counts[i]),int(prediction_counts[i])
        norm=float(np.linalg.norm(reference[i])*np.linalg.norm(prediction[i]))
        if nr and npred and np.isfinite(norm) and norm>1e-12:
            distance=float(1-np.clip(reference[i]@prediction[i]/norm,-1,1));reason=None
        else:
            distance=None
            reason='missing_reference_part' if not nr else 'missing_prediction_part' if not npred else 'degenerate_embedding'
        values[name]=distance
        details[name]=dict(distance=distance,reference_patches=nr,prediction_patches=npred,reason=reason)
    result=aggregate_visible_parts(values);result['parts']=details
    result['visibility_definition']='estimated confident skeleton support and own visible-foreground patch centres; not anatomical visibility ground truth'
    return result
