import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from helpers.tilePredictDtc_f import tilePredictDtc_f
from scipy.signal import medfilt2d
from skimage.measure import label
from helpers.tilePredictSgm_v2_f import tilePredictSgm_v2_f

test_opts = {'has_smpl_dstrb': 1, 'has_flip': 1, 'deform_magnitude': 0, 
            'deform_numb_grid_pnts': 1, 'offset_magnitude': [194, 282], 
            'rot_angle': 6.283185307179586, 'scale': [1, 1], 'netSize_1_l': [508, 324],'numb_aug_imgs': 1,
            'border_treat_mode': 1, 'disk_radius': 9, 'rp_id': 0, 'has_objcts_in_ignr_reg': False,
            'keep_prob': 0.5, 'netSize_2_l': [316, 132],
            'padding': 'zeros', 'average_mirror': False, 'path2save': '/home/haydemir/Desktop/connectivity/thesis/pythorch_port/fixed_tensor_3',
            'saveHDF5': True}

#set up intra-net skip connections
test_opts['expectedPredDataSgm'] = []
test_opts['expectedPredDataSgm'].append('n1_d1c')
test_opts['expectedPredDataSgm'].append('n1_d2c')
test_opts['expectedPredDataSgm'].append('n1_d3c')
test_opts['expectedPredDataSgm'].append('n1_d4c')
test_opts['expectedPredDataSgm'].append('obj_rp')

data_dir = '/home/haydemir/Desktop/connectivity/thesis/DiskMask/data_specs/c_elegans/example_data/ds_te/'
data_files = os.listdir(data_dir)

images_c = []
labels_c = []
weights_C = []
classes_c = []
file_names = []

#Load Dataset
for data_file in data_files:
    with h5py.File(data_dir + data_file, "r") as f:
        labels_c.append(np.transpose(f['labels'][()], (2, 1, 0))) #Get the values of the hdf5 labels as array
        images_c.append(np.transpose(f['data'][()], (2, 1, 0)))
        classes_c.append(np.squeeze(f['classes'][()]))
        file_names.append(data_file)

#process images
images_proc_c = []
gt_labs_c = []
gt_labs_sc_c = []
pred_labs_c = []
pred_labs_sc_c = []
pred_scores_c = []
soft_pred_labs_c = []
numb_data = len(images_c)

for im_i in range(numb_data):
    fileName_s = file_names[im_i][:-3]
    im = images_c[im_i]
    if im.shape[2] == 1:
        im = np.tile(im, (1, 1, 3))
    
    #predict reference points / object disks
    test_opts['padInput'] = np.array([test_opts['netSize_1_l'][0], test_opts['netSize_1_l'][0]])
    test_opts['padOutput'] = np.array([test_opts['netSize_1_l'][1], test_opts['netSize_1_l'][1]])
    test_opts['expectedPredData'] = []
    test_opts['expectedPredData'].append('pred_cls')

    scores = tilePredictDtc_f(im, test_opts)
    pred_cls = np.argmax(np.squeeze(scores[0]), axis=2)
    #pred_cls -= 1
    pred_cls = medfilt2d(pred_cls, [3, 3])
    pred_cls_cc = label(pred_cls > 0, connectivity=2)

    #predict segmentation masks based on object disks
    test_opts['expectedPredData'] = test_opts['expectedPredDataSgm']
    test_opts['padOutput'] = [test_opts['netSize_2_l'][1], test_opts['netSize_2_l'][1]]
    im_p = np.concatenate((im, pred_cls_cc[:, :, np.newaxis]), axis=2)
    scores = tilePredictSgm_v2_f(im_p, test_opts)

    #compute the objectness scores
    un_l = np.unique(pred_cls_cc)
    s_pred_l = []
    for ui in un_l:
        if ui == 0:
            continue
        s_m = scores[0][:, :, ui - 1].astype(float)
        #goodi = np.argwhere(s_m >= 0.5)
        s_pred_l.append(np.mean(s_m[s_m >= 0.5]))

    #store predictions
    soft_pred_labs_c.append(scores[0])
    pred_labs_c.append(scores[0] >= 0.5)
    pred_labs_sc_c.append(np.ones((1, pred_labs_c[-1].shape[2]), dtype=np.uint8))
    pred_scores_c.append(s_pred_l)
    images_proc_c.append(im)

    #save results
    path2saveres = '/home/haydemir/Desktop/connectivity/thesis/pythorch_port/inference_zerograd_continued'
    if test_opts['saveHDF5'] == True:
        path2hdf5 = path2saveres + '/' + fileName_s + '_imNr_' + str(im_i + 1) + '_' + str(numb_data) + '15000 + ''_predMasks.h5'

        masks_soft_pred = soft_pred_labs_c[-1]
        class_pred = pred_labs_sc_c[-1]
        obj_pred_scores = pred_scores_c[-1]

        hf = h5py.File(path2hdf5, 'w')

        hf.create_dataset('image', data=im)
        hf.create_dataset('masks_pred', data=(masks_soft_pred >= 0.5).astype(int))
        hf.create_dataset('masks_soft_pred', data=masks_soft_pred)
        hf.create_dataset('disk_mask_pred', data=pred_cls_cc.astype(np.uint16))
        hf.create_dataset('obj_pred_scores', data=obj_pred_scores)
        hf.create_dataset('class_pred', data=class_pred)

hf.close()




print(images_proc_c)