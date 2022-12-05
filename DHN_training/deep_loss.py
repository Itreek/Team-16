import numpy as np


def focal_loss(output, target, weights=None, gamma=2):

    if weights is None:
        loss = target * np.log(output) + (1 - target) * np.log(1 - output)
    else:
        loss = (np.pow(1 - output, gamma) * (target@weights[:, 1]) * np.log(output)) + (
            ((1 - target)@weights[:, 0]) * np.log(1.0 - output)*np.pow(output, gamma))

    final_loss = -np.mean(loss)

    return final_loss
