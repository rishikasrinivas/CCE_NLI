import numpy as np

def iou(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / (union + np.finfo(np.float32).tiny)

def detection_acc(a,b):
    intersection = (a & b).sum()
    return intersection/b.sum()