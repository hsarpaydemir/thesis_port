import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt

def cropBlob_f(data_c, netSize_l, isNetIn=False):
    
    if data_c.shape[0] != data_c.shape[1]:
        ValueError('size(data_c, 1) must be equal to size(data_c,2)')
    if data_c.shape[0] > netSize_l[0]:
        ValueError('size(data_c, 1) must be smaller than netSize_l(1)')
    
    padsize = 0
    if data_c.shape[0] != netSize_l[0]:
        padsize = np.abs(netSize_l[0] - data_c.shape[0]) #random translation is based on the netoutput => pad the image
    if padsize % 2 !=  0:
        ValueError('padsize must be even')
    else:
        padsize = padsize / 2 #padsize must be even

    if padsize != 0:
        data_c = np.pad(data_c, padsize, mode='constant', constant_values=0)
    if isNetIn == False:
        data_c = cropIt2NetS_loc(data_c, netSize_l)
    
    return data_c

#helper functions
def cropIt2NetS_loc(data, netSize_l):
    #compute the coordinates of the frame in the middle
    labCropRect = np.array([(netSize_l[0] - netSize_l[1]) / 2 + 1, (netSize_l[0] - netSize_l[1]) / 2 + 1, (netSize_l[0] - netSize_l[1]) / 2 + netSize_l[1], (netSize_l[0] - netSize_l[1]) / 2 + netSize_l[1]], dtype=int)
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis, np.newaxis]
    elif len(data.shape) == 3:
        data = data[:, :, :, np.newaxis]
    croppedData = np.zeros((netSize_l[1], netSize_l[1], data.shape[2], data.shape[3]), like=np.array(data))

    for ds_i in range(data.shape[3]):
        croppedData[:, :, :, ds_i] = applyOpChannelWise(myImcrop, data[:, :, :, ds_i], croppedData[:, :, :, ds_i], labCropRect)
    
    return croppedData

def applyOpChannelWise(func, A, B, labCropRect):
    func_bu = func
    for ch_i in range(A.shape[2]):
        func = func_bu
        B[:, :, ch_i] = func(A[:, :, ch_i], labCropRect)
    return B

def myImcrop(im, labCropRect):
    im = im[labCropRect[0]:labCropRect[2] + 1, labCropRect[1]:labCropRect[3] + 1]
    return im