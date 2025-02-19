import numpy as np

def iou(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / (union + np.finfo(np.float32).tiny)

def samples_coverage(acts,formula):
    intersection = (acts & formula).sum()
    num_samples_active_for_form=formula.sum()
    return intersection, num_samples_active_for_form, intersection/num_samples_active_for_form
                
def explanation_coverage(acts,formula):
    intersection = (acts & formula).sum()
    num_active_in_range=acts.sum()
    return intersection, num_active_in_range, intersection/num_active_in_range