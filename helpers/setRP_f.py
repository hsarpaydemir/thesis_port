from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt

def setRP_f(masksRPs_c, rp_id):
    #places Reference Point (RP) relative to instance mask (in "masksRPs_c")
    #The RP's position can be defined by "RFlab": 
    # - rp_id==0: set RP to the object centroid,
    #- rp_id==-1: set RP to bounding boxes (not tested),
    # - rp_id > 0: to the centroid of the class "rp_id" in the object segmentation

    maskStack = masksRPs_c
    for ch_i in range(maskStack.shape[2]):
        tmp_maskStackSl = maskStack[:,:,ch_i].astype(int)
        if rp_id == 0:
            tmp_maskStackSl = (tmp_maskStackSl > 0).astype(int)
            stat_lab = regionprops(tmp_maskStackSl)
        #rp == -1 is not tested therefore not ported
        elif rp_id > 0:
            tmp_maskStackSl = tmp_maskStackSl == rp_id
            if np.all(tmp_maskStackSl == 0):
                tmp_maskStackSl = tmp_maskStackSl > 0
            tmp_maskStackSl = tmp_maskStackSl.astype(int)
            stat_lab = regionprops(tmp_maskStackSl)
        
        tmp_maskStackSl = np.zeros(tmp_maskStackSl.shape, dtype=int)
        if len(stat_lab) != 0:
            rp = np.array([round(stat_lab[0].centroid[0]), round(stat_lab[0].centroid[1])])
            rp[rp < 1] = 1
            if rp[0] > tmp_maskStackSl.shape[0]:
                rp[0] = tmp_maskStackSl.shape[0]
            if rp[1] > tmp_maskStackSl.shape[1]:
                rp[1] = tmp_maskStackSl.shape[1]
            tmp_maskStackSl[rp[0], rp[1]] = 1
        maskStack[:, :, ch_i] = tmp_maskStackSl
    masksRPs_c = maskStack

    return masksRPs_c
            
