from helpers.precomputeSFD_f import precomputeSFD_f
from helpers.ignoreMasksOnBorders_f import ignoreMasksOnBorders_f
from helpers.genDsksOfadaptSize_f import genDsksOfadaptSize_f
from helpers.weightSpaceBtwDsks_f import weightSpaceBtwDsks_f
from helpers.cropBlob_f import cropBlob_f
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import torch

def prepareData_f(train_dataloader, augment_opts):
    #parameters for weighting space btw. object disks
    NUMB_AUG_IMGS = 10

    sigma_px = 3.535531
    w_0 = 15

    augData_c = []
    augLabs_c = []
    augWeights_c = []

    tmp_augLabs_c = []
    tmp_augClasses_c = []
    tmp_augWeights_c = []
    for aim_i in range(NUMB_AUG_IMGS):
        #augment data, data will be croped to the net input
        augData, tmp_augL, tmp_augC, augWeight, img_name = next(iter(train_dataloader)).values()

        #Getting rid of the batch size dimension
        augData, tmp_augL, tmp_augC, augWeight = torch.permute(augData.squeeze(), (1, 2, 0)), torch.permute(tmp_augL.squeeze(), (1, 2, 0)), tmp_augC.squeeze(), augWeight.squeeze()
        augData_c.append(augData)
        tmp_augLabs_c.append(tmp_augL)
        tmp_augClasses_c.append(tmp_augC)
        tmp_augWeights_c.append(augWeight)
        
        #find and ignore the instances on the borders (input tile), since
        #their centroids can not be computed correctly.
        tmp_augL, tmp_augC, ignoreOB = ignoreMasksOnBorders_f(tmp_augL, tmp_augC)
        tmp_augLabs_c[aim_i] = tmp_augL
        tmp_augClasses_c[aim_i] = tmp_augC

        #set Reference Points (RP) and compute their size
        dskswCl, rpIDs, rpWgtsInsDsks = genDsksOfadaptSize_f(tmp_augLabs_c[aim_i], tmp_augClasses_c[aim_i], augment_opts['disk_radius'], augment_opts['rp_id'])
        dskswCl[dskswCl == -1] = 0
        rpIDs += 1
        
        #weight space btw. object disks
        rpWgtsBtwDsks = weightSpaceBtwDsks_f(dskswCl, sigma_px, w_0)
        rpWgtsBtwDsks[dskswCl > 0] = 0 #set to zero at object disks

        #putting weights together   
        ignoreMask = np.logical_or(np.logical_or(tmp_augWeights_c[aim_i] == 0, dskswCl == 255), np.array(ignoreOB, dtype=bool))
        wghtCls = np.ones(dskswCl.shape)
        wghtCls = wghtCls + rpWgtsInsDsks + rpWgtsBtwDsks
        wghtCls[np.array(ignoreMask, dtype=bool)] = 0

        # crop sLabels_c and sWeights_c
        augLabs_c1 = cropBlob_f(dskswCl, augment_opts['netSize_1_l'])
        augWeights_c.append(np.squeeze(cropBlob_f(wghtCls, augment_opts['netSize_1_l'])))
        augLabs_c2 = cropBlob_f(rpIDs, augment_opts['netSize_1_l'])
        augLabs_c3 = cropBlob_f(tmp_augLabs_c[aim_i], augment_opts['netSize_1_l'])
        augLabs_c.append([np.squeeze(augLabs_c1), np.squeeze(augLabs_c2), np.squeeze(augLabs_c3)])

    print('end of prepare_data.f')
    
    return augData_c, augLabs_c, augWeights_c