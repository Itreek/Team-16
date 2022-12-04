import numpy as np
import pandas as pd
from distances import Find_IoU
from mot import MOTAccumulator
from metrics import num_frames

def load_mot(fname):
    df = pd.read_csv(fname,skipinitialspace=True,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],engine='python')
    del df['unused']
    return df

gt = pd.read_csv('gt.txt',skipinitialspace=True,names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'],
        engine='python'
    )
    
det = load_mot('det.txt')

def get_columns_per_frame(df,num_frames):
    number_of_objects_per_frame = [[] for i in range(num_frames)]
    for i in range(1,num_frames+1):
        object_idxs = np.where(df.FrameId == i)
        number_of_objects_per_frame[i-1].append(df.loc[object_idxs[0],['X', 'Y','Width','Height']])
    return number_of_objects_per_frame

gt_frames = num_frames(gt)
det_frames = num_frames(det)
columns_per_frame_gt = get_columns_per_frame(gt,gt_frames)
columns_per_frame_det = get_columns_per_frame(det,det_frames)

def CLEAR_MOT(gt, dt,include_all=False, vflag=''):

    compute_dist = Find_IoU(columns_per_frame_gt[:,0],columns_per_frame_det[:,0],0.5)
    
    assignment_mat = MOTAccumulator()
    
    #considers only object instances above confidence 0.99
    if include_all:
        gt = gt[gt['Confidence'] >= 0.99]
    else:
        gt = gt[(gt['Confidence'] >= 0.99) & (gt['ClassId'] == 1)]
        
    allframeids = gt.index.union(dt.index).levels[0]
    analysis = {'hyp': {}, 'obj': {}}
    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))

        if fid in gt.index:
            fgt = gt.loc[fid]
            oids = fgt.index.values
            for oid in oids:
                oid = int(oid)
                if oid not in analysis['obj']:
                    analysis['obj'][oid] = 0
                analysis['obj'][oid] += 1

        if fid in dt.index:
            fdt = dt.loc[fid]
            hids = fdt.index.values
            for hid in hids:
                hid = int(hid)
                if hid not in analysis['hyp']:
                    analysis['hyp'][hid] = 0
                analysis['hyp'][hid] += 1

        if oids.shape[0] > 0 and hids.shape[0] > 0:
            dists = compute_dist(fgt[['X', 'Y', 'Width', 'Height']].values, fdt[['X', 'Y', 'Width', 'Height']].values)

        assignment_mat.update(oids, hids, dists, frameid=fid, vf=vflag)

    return assignment_mat, analysis



