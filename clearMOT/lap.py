# Linear assignment problem
import numpy as np
from munkres import Munkres

def get_padded_square(inputs):
    nr, nc = inputs.shape[0],inputs.shape[1]
    if np.equal(nr,nc):
        return inputs
    pad_inputs = np.zeros((np.max(nr,nc), np.max(nr,nc)), dtype=inputs.dtype)
    pad_inputs[:nr, :nc] = inputs
    return pad_inputs

def add_edges(costs):
    if np.all(np.isfinite(costs)):
        return costs.copy()
    if not np.any(np.isfinite(costs)):
        return np.zeros_like(costs)
    x1 = np.min(costs.shape)
    x2 = np.max(np.abs(costs[np.isfinite(costs)])) + 1 
    output = np.where(np.isfinite(costs), costs,(2*x1*x2 + 1))
    return output

def remove_edges(costs, row_ids, col_ids):
    sub = [
        idx for idx, (i, j) in enumerate(zip(row_ids, col_ids))
        if np.isfinite(costs[i, j])
    ]
    return row_ids[sub], col_ids[sub]

def munkres_solver(costs):
    model = Munkres()
    new_costs = add_edges(costs)
    new_costs = get_padded_square(new_costs)
    indices = np.array(model.compute(new_costs), dtype=int)
    indices = indices[(indices[:, 0] < costs.shape[0]) & (indices[:, 1] < costs.shape[1])]
    row_ids, col_ids = indices[:, 0], indices[:, 1]
    row_ids, col_ids = remove_edges(costs, row_ids, col_ids)
    return row_ids, col_ids

def linear_sum_assignment(costs):
    costs = np.asarray(costs)
    if not costs.size:
        return np.array([], dtype=int), np.array([], dtype=int)
    row_ids, col_ids = munkres_solver(costs)
    row_ids ,col_ids = int(np.asarray(row_ids)), int(np.asarray(col_ids))
    return row_ids, col_ids