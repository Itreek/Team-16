import numpy as np
from distances import Find_IoU
from lap import linear_sum_assignment

def preprocessResult(res, gt, inifile):
    labels = [
        'ped',               # 1
        'person_on_vhcl',    # 2
        'car',               # 3
        'bicycle',           # 4
        'mbike',             # 5
        'non_mot_vhcl',      # 6
        'static_person',     # 7
        'distractor',        # 8
        'occluder_on_grnd',  # 9
        'occluder_full',     # 10
        'reflection'         # 11
    ]
    dicts = [ 'static_person', 'person_on_vhcl', 'distractor', 'reflection']
   
   