from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt
import scipy.io as sio

def weightSpaceBtwDsks_f(dskswCl, sigma_px, w_0):
    ccMasks = label(dskswCl, connectivity=2)
    inds = np.unique(ccMasks)
    inds = np.delete(inds, 0)
    minDist1 = 1e10 * np.ones(ccMasks.shape)
    minDist2 = 1e10 * np.ones(ccMasks.shape)
    for li in range(len(inds)):
        tmp_bwMask = (ccMasks == inds[li])
        dist = distance_transform_edt(1 - tmp_bwMask)
        minDist2 = np.minimum(minDist2, dist)
        newMin1  = np.minimum(minDist1, minDist2)
        newMin2  = np.maximum(minDist1, minDist2)
        minDist1 = newMin1.copy()
        minDist2 = newMin2.copy()

    rpWgtsBtwDsks = np.array((np.exp(-(minDist1 + minDist2) ** 2 / (2 * sigma_px ** 2))) * w_0, dtype=np.single)

    return rpWgtsBtwDsks