import numpy as np   

def Min_Max_BoundingBoxes(elem):
    min_part = elem[..., :2]
    size = elem[..., 2:]
    max_part = min_part + size
    return min_part, max_part

def Find_IoU(n1, n2):
    #Computes Intersection of Union of two rectangular bounding boxes
    n1_min, n1_max = Min_Max_BoundingBoxes(n1)
    n2_min, n2_max = Min_Max_BoundingBoxes(n2)
    # Compute intersection.
    intersection_max,intersection_min = np.minimum(n1_max, n2_max), np.maximum(n1_min, n2_min)
    intersection_size = np.maximum(intersection_max - intersection_min, 0)
    intersection_volume = np.prod(intersection_size, axis=-1)
    # Computes volume of the union.
    n1_size,n2_size = np.maximum(0,n1_max - n1_min),np.maximum(0,n2_max - n2_min) 
    n1_volume, n2_volume = np.prod(n1_size, axis=-1), np.prod(n2_size, axis=-1)
    u_volume = n1_volume + n2_volume - intersection_volume
    return np.where(intersection_volume == 0, np.zeros_like(intersection_volume, dtype=np.float64),np.true_divide(intersection_volume, u_volume))


def IoU_matrix(gt_ids, det_ids, max_possible_iou=1):
    if np.size(det_ids) == 0 or np.size(gt_ids) == 0:
        return np.empty((0, 0))
    det_ids = np.asfarray(det_ids)
    gt_ids = np.asfarray(gt_ids)    
    iou = Find_IoU(gt_ids[:, None], det_ids[None, :])
    dist = 1 - iou
    dist_new = np.where(dist > max_possible_iou, np.NaN, dist)
    return dist_new