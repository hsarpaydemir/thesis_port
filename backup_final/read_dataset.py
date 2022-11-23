from html.entities import name2codepoint
import h5py
import numpy as np
from PIL import Image
from skimage.draw import polygon
import os
import matplotlib.pyplot as plt
import random

filename = "/home/haydemir/Desktop/connectivity/thesis/pythorch_port/inference_zerograd_continued/C17_1_15_1_mod_imNr_2_215000 + _predMasks.h5"

class_pred = []
disk_mask_pred = []
my_image = []
masks_pred = []
masks_soft_pred = []
obj_pred_scores = []

with h5py.File(filename, "r") as f:
    class_pred.append(np.array(f['class_pred'][()]))
    disk_mask_pred.append(np.array(f['disk_mask_pred'][()]))
    my_image.append(np.array(f['image'][()]))
    masks_pred.append(np.array(f['masks_pred'][()]))
    masks_soft_pred.append(np.array(f['masks_soft_pred'][()]))
    obj_pred_scores.append(np.array(f['obj_pred_scores'][()]))

for i in range(np.squeeze(masks_pred[0]).shape[2]):
    my_image[0][:, :, 0:1] = my_image[0][:, :, 0:1] + random.random() * np.squeeze(masks_pred[0])[:, :, i:i + 1]
    my_image[0][:, :, 1:2] = my_image[0][:, :, 1:2] + random.random() * np.squeeze(masks_pred[0])[:, :, i:i + 1]
    my_image[0][:, :, 2:3] = my_image[0][:, :, 2:3] + random.random() * np.squeeze(masks_pred[0])[:, :, i:i + 1]

print(my_image.shape)
'''
data_dir = dir([opts.path2TrH5Files '*.h5']);
for f_i = 1 : numel(data_dir); images_c{f_i} = hdf5read([data_dir(f_i).folder '/' data_dir(f_i).name],'/data'); end
if strcmp(opts.trte,'tr')
    for f_i = 1 : numel(data_dir); labels_c{f_i} = hdf5read([data_dir(f_i).folder '/' data_dir(f_i).name], '/labels'); end
    for f_i = 1 : numel(data_dir); weights_c{f_i} = single(hdf5read([data_dir(f_i).folder '/' data_dir(f_i).name], '/ignore') == 0); end
    for f_i = 1 : numel(data_dir); classes_c{f_i} = hdf5read([data_dir(f_i).folder '/' data_dir(f_i).name], '/classes'); end
    sfD_c = precomputeSFD_f(labels_c, weights_c);
end
'''


"""
sample_2 = Image.open("../sample_images/Test_2.tif")
sample_2_array = np.array(sample_2)
joint_roi_map = np.zeros(sample_2_array.shape)

rois = read_roi_zip("../RoiSettest2final.zip")

for name_of_mask in rois.keys():
    #Transform the segmentation maps to array format
    '''
    for i in range(len(rois[name_of_mask]['x'])):
        mask_x = rois[name_of_mask]['x'][i]
        mask_y = rois[name_of_mask]['y'][i]
        if mask_x > 1316:
            mask_x = 1316
        if mask_y > 992:
            mask_y = 992

        joint_roi_map[mask_y][mask_x] = 255
    '''
    
    #Fill in the segmentation masks with polygon
    x_array = np.array(rois[name_of_mask]['x'])
    y_array = np.array(rois[name_of_mask]['y'])

    rr, cc = polygon(y_array, x_array)
    joint_roi_map[rr, cc] = 255

masked_image = Image.fromarray(joint_roi_map)
masked_image.show()

print(sample_2_array.shape)
#sample_2.show()
"""

'''
def prepareData_f(sData_c, sLabels_c, sClasses_c, sDiffData_c, sWeights_c, sfDs_c, opts):
    if not hasattr(opts, 'rp_id'):
        opts.rp_id = 0

    #parameters for weighting space btw. object disks
    sigma_px = 3.535531
    w_0 = 15

    augData_c = []
    augLabs_c = []
    augWeights_c = []
    augDiff_c = []

    tmp_augLabs_c = []
    tmp_augClasses_c = []
    tmp_augWeights_c = []

    numb_data = np.prod(sData_c.shape)
    for aim_i in range(1, len(opts.numb_aug_imgs)):
        np.random.randint(low=1, high=numb_data + 1)
        augData_c(aim_i), tmp_augLabs_c(aim_i), tmp_augClasses_c(aim_i), tmp_augWeights_c(aim_i), augDiff_c(:,aim_i) = augmentData_v2_f(sData_c(rnd_i), sLabels_c(rnd_i), sClasses_c(rnd_i), sWeights_c(rnd_i), sDiffData_c(:,rnd_i),1, sfDs_c(rnd_i), opts)

    return augData_c, augLabs_c, augWeights_c, augDiff_c
'''

'''
def augmentData_v2_f(sData_c, sLabels_c, sClasses_c, origWeights_c, sDiffData_c, numb_aug_imgs, sfD_c, opts):
    
    return augData_c, augLabels_c, augClasses_c, augWeights_c, augDiffData_c
'''