from skimage.morphology import diamond, disk
from skimage.morphology import footprints
from helpers.setRP_f import setRP_f
import scipy.io as sio
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import cv2
import math
import matplotlib.pyplot as plt
import torch

def genDsksOfadaptSize_f(sMask_c, augClass_c, disk_radius, rp_id):
    #outputs reference points with ID ("mIDRPs") where IDs == #channel of the corresponding instance mask.
    #"mRPs" provides disks with obejct classes. "tWghts" provides weights.
    #note: object class > 1 is not tested.

    se = diamond(radius=3)
    m_RP_cl = augClass_c
    mRPs = setRP_f(sMask_c, rp_id)
    #mRPs2 = sio.loadmat('mRPs.mat')['mRPs']

    mRPs = np.argmax(np.concatenate(((np.ones(mRPs[:, :, 0].shape, dtype=int) * 0.1)[:, :, np.newaxis], mRPs), axis=2), axis=2)
    mRPs -= 1
    mIDRPs = mRPs.copy()
    m_RPD, m_RPDind = distance_transform_edt(1 - np.where(mRPs + 1 > 0, 1, 0), return_indices=True)
    m_RPDind = m_RPDind[1] * m_RPD.shape[1] + m_RPDind[0]
    tWghts = np.zeros(mRPs.shape)
    rp_ps = np.unique(m_RPDind)

    for d_i in rp_ps:
        tmp_bwMask = (m_RPDind == d_i)
        tmp_erMask = cv2.erode(np.array(tmp_bwMask, dtype=np.uint8), se)
        tmp_diffMask = tmp_bwMask - tmp_erMask
        if (len(m_RPD[tmp_diffMask == 1])) > 0:
            disk_radius_n = max(float(min(math.floor(min(m_RPD[tmp_diffMask == 1])), disk_radius)), 0)
        else:
            disk_radius_n = disk_radius
        
        if disk_radius_n <= 2:
            se_d = disk(disk_radius_n)
        else:
            se_d = footprints.ellipse(int(disk_radius_n) - 1, int(disk_radius_n) - 1)
        tmp_dltMask = np.zeros(mRPs.shape, dtype=int)
        tmp_dltMask[np.unravel_index(d_i, tmp_dltMask.shape, 'F')] = int(mRPs[np.unravel_index(d_i, mRPs.shape, 'F')] >= 0)
        tmp_dltMask = cv2.dilate(np.array(tmp_dltMask, dtype=np.uint8), se_d)
        tmp_dltnMask = np.array(tmp_dltMask, dtype=bool)
        cl = mRPs[np.unravel_index(d_i, mRPs.shape, 'F')]
        if len(m_RP_cl.shape) > 1:
            m_RP_cl = np.squeeze(m_RP_cl)
        cl = m_RP_cl[cl]
        mRPs[tmp_dltnMask] = tmp_dltMask[tmp_dltnMask] * int(cl)
        wght_tmp = disk_radius - disk_radius_n + 1
        tWghts[tmp_dltnMask] = tWghts[tmp_dltnMask] + wght_tmp


    return mRPs, mIDRPs, tWghts