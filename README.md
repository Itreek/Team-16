# Commands to run CLEAR-MOT for a sample test case
##### Go to the terminal in which python is running and run the following commands 

import motmetrics as mm
import numpy as np
acc = mm.MOTAccumulator(auto_id=True)
acc.update(
    [1, 2],                     # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
    ]
)

### next - To view the accumulator's events 
print(acc.events)
frameid = acc.update(
    [1, 2], # the ground-truth obj
    [1],  # the predicted obj
    [
        [0.2], 
        [0.4]
    ] # distance matrix
)
print(acc.mot_events.loc[frameid])

### next - To view the MOT metrics
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp'], name='acc')
print(summary)
