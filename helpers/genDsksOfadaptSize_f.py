from skimage.morphology import diamond, disk
from .setRP_f import setRP_f
import scipy.io as sio
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import cv2
import math

def genDsksOfadaptSize_f(sMask_c, augClass_c, disk_radius, rp_id):
    #outputs reference points with ID ("mIDRPs") where IDs == #channel of the corresponding instance mask.
    #"mRPs" provides disks with obejct classes. "tWghts" provides weights.
    #note: object class > 1 is not tested.

    se = diamond(radius=3)
    m_RP_cl = augClass_c
    mRPs = setRP_f(sMask_c, rp_id)
    mRPs2 = sio.loadmat('mRPs.mat')['mRPs']

    mRPs = np.argmax(np.concatenate(((np.ones(mRPs[:, :, 0].shape, dtype=int) * 0.1)[:, :, np.newaxis], mRPs), axis=2), axis=2)
    mRPs -= 1
    mIDRPs = mRPs
    m_RPD, m_RPDind = distance_transform_edt(1 - np.where(mRPs + 1 > 0, 1, 0), return_indices=True)
    m_RPDind = m_RPDind[1] * m_RPD.shape[1] + m_RPDind[0]
    tWghts = np.zeros(mRPs.shape)
    rp_ps = np.unique(m_RPDind)

    for d_i in rp_ps:
        tmp_bwMask = (m_RPDind == d_i)
        tmp_erMask = cv2.erode(np.array(tmp_bwMask, dtype=np.uint8), se)
        tmp_diffMask = tmp_bwMask - tmp_erMask
        disk_radius_n = max(float(min(math.floor(min(m_RPD[tmp_diffMask == 1])), disk_radius)), 0)
        if not disk_radius_n:
            disk_radius_n = disk_radius
        
        se_d = disk(disk_radius_n)
        tmp_dltMask = np.zeros(mRPs.shape, dtype=int)
        tmp_dltMask.ravel()[d_i] = int(mRPs.ravel()[d_i] > 0)

    print(mRPs)