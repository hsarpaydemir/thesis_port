import numpy as np
import cv2
import os
import h5py
import skimage

def precomputeSFD_f(sLabels_c, sWeights_c):
    sfD_c = []

    for l_i in range(len(sLabels_c)):
        tmp_sfD = sLabels_c[l_i]
        tmp_sfD[tmp_sfD == 255] = 0
        tmp_sfD = np.sum(tmp_sfD, 2)
        tmp_sfD = np.array(np.greater(tmp_sfD, 0) & np.not_equal(sWeights_c[l_i], 0)).astype('float32')
        #tmp_sfD = cv2.GaussianBlur(tmp_sfD, ksize=(0, 0), sigmaX=50, borderType=cv2.BORDER_REPLICATE)
        tmp_sfD = skimage.filters.gaussian(tmp_sfD, sigma=50,mode = 'nearest',truncate=2.0)
        m_val = np.amax(tmp_sfD)
        if m_val < 0.0303:
            m_val = 0.0303
        tmp_sfD = tmp_sfD / m_val
        sfD_c.append(np.array(tmp_sfD).astype('float32'))
    return sfD_c

'''
data_dir = '/home/haydemir/Desktop/connectivity/thesis/DiskMask/data_specs/c_elegans/example_data/ds_tr/'
data_files = os.listdir(data_dir)

labels_c = []
weights_c = []
for data_file in data_files:
    with h5py.File(data_dir + data_file, "r") as f:
        labels_c.append(np.transpose(f['labels'][()], (2, 1, 0))) #Get the labels of the hdf5 dataset as array
        weights_c.append((np.transpose(f['ignore'][()][0, :, :]) == 0).astype('float32')) #Load the weights with the same form as in matlab

sfD_c = precomputeSFD_f(labels_c, weights_c)
print(labels_c)
'''
