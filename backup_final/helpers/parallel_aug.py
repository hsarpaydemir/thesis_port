import multiprocessing
from loadData import loadData
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
import scipy.io as sio
from skimage.morphology import thin
import random
import time

def temp_fnc():
    NUMB_AUG_IMGS = 10

    sigma_px = 3.535531
    w_0 = 15

    augData_c = []
    augLabs_c = []
    augWeights_c = []

    tmp_augLabs_c = []
    tmp_augClasses_c = []
    tmp_augWeights_c = []

    #augment data, data will be croped to the net input
    iter_time = time.time()
    augData, tmp_augL, tmp_augC, augWeight, img_name = next(iter(train_dataloader)).values()
    iter_time2 = time.time()
    print('Iterate time: ', iter_time2 - iter_time)

    #Getting rid of the batch size dimension
    augData, tmp_augL, tmp_augC, augWeight = np.squeeze(augData), np.squeeze(tmp_augL), np.squeeze(tmp_augC), np.squeeze(augWeight)
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

if __name__ == '__main__':
    NUMB_WORKERS = 10
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size)

    results = []
    train_dataloader = DataLoader(elegans_dataset, batch_size=1, sampler=torch.utils.data.RandomSampler(elegans_dataset, replacement=True))

    result = pool.map_async(prepareData_f, range(NUMB_WORKERS)).get()

    pool.close()
    pool.join()
    print('stopping here for parallel checking')

    augment_opts = {'has_smpl_dstrb': 1, 'has_flip': 1, 'deform_magnitude': 0, 
                    'deform_numb_grid_pnts': 1, 'offset_magnitude': [194, 282], 
                    'rot_angle': 6.283185307179586, 'scale': [1, 1], 'netSize_1_l': [508, 324],'numb_aug_imgs': 1,
                    'border_treat_mode': 1, 'disk_radius': 9, 'rp_id': 0, 'has_objcts_in_ignr_reg': False,
                    'keep_prob': 0.5, 'netSize_2_l': [316, 132]}

