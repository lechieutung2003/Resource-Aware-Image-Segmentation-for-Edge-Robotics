import numpy as np

def calculate_iou(pred, target):
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update

    # TODO: optimize this block
