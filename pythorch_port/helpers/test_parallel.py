import multiprocessing
from loadData import loadData
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from detectRPsOnBorders_f import detectRPsOnBorders_f
from activateRPandMask_v2_f import activateRPandMask_v2_f
import scipy.io as sio
from skimage.morphology import thin
import random
from cropBlob_f import cropBlob_f

if __name__ == '__main__':
    NUMB_WORKERS = 5
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size)

    results = []

    result = pool.map_async(loadData, range(NUMB_WORKERS)).get()
    
    pool.close()
    pool.join()
    print('stopping here for parallel checking')

    augment_opts = {'has_smpl_dstrb': 1, 'has_flip': 1, 'deform_magnitude': 0, 
                    'deform_numb_grid_pnts': 1, 'offset_magnitude': [194, 282], 
                    'rot_angle': 6.283185307179586, 'scale': [1, 1], 'netSize_1_l': [508, 324],'numb_aug_imgs': 1,
                    'border_treat_mode': 1, 'disk_radius': 9, 'rp_id': 0, 'has_objcts_in_ignr_reg': False,
                    'keep_prob': 0.5, 'netSize_2_l': [316, 132]}


    #matlab_params = sio.loadmat('detectRPsOnBorders.mat')
    for r_w in range(NUMB_WORKERS):
        #augData_c = matlab_params['augData_c'][r_w][0]
        #augLabs_c = matlab_params['augLabs_c'][r_w][0]
        #augWeights_c = matlab_params['augWeights_c'][r_w][0]
        augData_c = result[r_w][1]
        augLabs_c = result[r_w][2]
        augWeights_c = result[r_w][3]
        for r_d in range(len(result[r_w][1])):
            #im = augData_c[r_d][0]
            im = augData_c[r_d]
            if im.shape[2] == 1:
                im = np.tile(im, (1, 1, 3))
            #Need to connect the input of enconet('data' in caffe) to im here

            #
            #Set up EncoNet
            #
            dskswCl = augLabs_c[r_d][0].copy()
            wghtCls = augWeights_c[r_d].copy()
            ignore = wghtCls == 0

            #handle segmentations in ignore regions
            if augment_opts['has_objcts_in_ignr_reg']:
                wghtCls[np.logical_and((dskswCl > 0), (wghtCls == 0))] = 1
                ignore_rp = wghtCls == 0
                dskswCl[ignore_rp] = 255
            else:
                dskswCl[ignore] = 255
            
            #Need to connect the input of enconet('label_2D_cls') to dskswCl here
            #Need to connect the input of enconet('weight_2D_cls') to wghtCls here

            #Set up gateways
            augLabs = augLabs_c[r_d][2]
            bwDskswCl = augLabs_c[r_d][0] > 0
            rpIDs = augLabs_c[r_d][1]
            ccDsks = label(bwDskswCl, connectivity=2)

            #find RPs on tile borders
            rp_r2k_m, ref2reject = detectRPsOnBorders_f(rpIDs, ccDsks)
            
            #activate a set of gateways,
            #keep gateways on borders closed, so we can augment the disks by thinning operation
            augLabs, bwDskswCl = activateRPandMask_v2_f(rpIDs, augLabs, ccDsks, bwDskswCl, ref2reject, augment_opts['keep_prob'])

            #augment gateway size
            bwDskswCl = thin(bwDskswCl, random.randint(1, augment_opts['disk_radius'] - 1) - 1)
            #Need to connect the input of enconet('gates') to bwDskswCl here

            #
            #set up DecoNet
            #
            #ignore subclasses, project to 2D
            augLabs = np.amax(np.array(augLabs > 0, dtype=np.uint8), axis=2)

            #crop to the DecoNet output
            ignore = np.squeeze(cropBlob_f(ignore, [augment_opts['netSize_1_l'][1], augment_opts['netSize_2_l'][1]]))
            augLabs = np.squeeze(cropBlob_f(augLabs, [augment_opts['netSize_1_l'][1], augment_opts['netSize_2_l'][1]]))

            #handle segmentations in ignore regions
            if augment_opts['has_objcts_in_ignr_reg']:
                ignore[np.logical_and(augLabs > 0, ignore == 1)] = 0
            augLabs[ignore.astype(int)] = 255
            #Need to connect the input of enconet('gt_semseg') to augLabs here
            
            #track loss

            print(rp_r2k_m.shape)