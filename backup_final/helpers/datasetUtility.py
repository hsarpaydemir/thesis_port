from __future__ import print_function, division
from curses import termattrs
from distutils.log import error
from operator import index
import os
from signal import siginterrupt
from stat import SF_APPEND
from telnetlib import TM
from turtle import up
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
import random
from torchvision.io import read_image
import math
from PIL import Image
from scipy.interpolate import interpn, interp2d, LinearNDInterpolator, RectBivariateSpline, SmoothBivariateSpline, LSQBivariateSpline, griddata, NearestNDInterpolator
from scipy.ndimage import map_coordinates
from helpers.precomputeSFD_f import precomputeSFD_f
from helpers.ignoreMasksOnBorders_f import ignoreMasksOnBorders_f
from helpers.genDsksOfadaptSize_f import genDsksOfadaptSize_f
from helpers.weightSpaceBtwDsks_f import weightSpaceBtwDsks_f
from helpers.cropBlob_f import cropBlob_f
from helpers.prepareData_f import prepareData_f
from matplotlib import pyplot as plt
import scipy.io as sio
from tqdm import tqdm
import multiprocessing
import time

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Custom compose class to accept 2 arguments to the __call__ function of AugmentData_v2_f
class MyCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        if target is not None:
            for t in self.transforms:
                img = t(img, target)
            return img
        else:
            for t in self.transforms:
                img = t(img)
            return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class ElegansDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        data_files = os.listdir(data_dir)
        self.img_labels = []
        self.images_c= []

        labels_c = []
        weights_c = []
        classes_c = []
        self.img_names = []
        for data_file in data_files:
            with h5py.File(data_dir + data_file, "r") as f:
                self.img_labels.append(f['labels'][()]) #Get the values of the hdf5 labels as array
                self.images_c.append(f['data'][()])

                labels_c.append(np.transpose(f['labels'][()], (2, 1, 0))) #Get the labels of the hdf5 dataset as array
                weights_c.append((np.transpose(f['ignore'][()][0, :, :]) == 0).astype('float32')) #Load the weights with the same form as in matlab
                classes_c.append(np.array(f['classes'][()], dtype=int))
                self.img_names.append(data_file)

        #Randomly permuting the images and respective arrays
        batchSfD_c = precomputeSFD_f(labels_c, weights_c)
        batchIms_c = self.images_c 
        batchLabs_c = self.img_labels
        batchWghts_c = weights_c
        batchCls_c = classes_c
        permute_temp = list(zip(batchIms_c, batchLabs_c, batchCls_c, batchWghts_c, batchSfD_c, self.img_names))
        random.shuffle(permute_temp)
        self.batchIms_c, self.batchLabs_c, self.batchCls_c, self.batchWghts_c, self.batchSfD_c, self.img_names = zip(*permute_temp)

        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        sample = {'batchIms_c': self.batchIms_c[idx], 'batchLabs_c': self.batchLabs_c[idx], 
        'batchCls_c': self.batchCls_c[idx], 'batchWghts_c': self.batchWghts_c[idx], 'img_name': self.img_names[idx]}
        if self.transform:
            sample['batchIms_c'], sample['batchLabs_c'], sample['batchWghts_c'] = self.transform(self.batchIms_c[idx], self.batchLabs_c[idx], self.batchWghts_c[idx])
        return sample

class AugmentData_v2_f(object):
    def __init__(self, opts):
        self.has_smpl_dstrb = opts['has_smpl_dstrb']
        self.has_flip = opts['has_flip']
        self.deform_magnitude = opts['deform_magnitude']
        self.deform_numb_grid_pnts = opts['deform_numb_grid_pnts']
        self.offset_magnitude = opts['offset_magnitude']
        self.scale = opts['scale']
        self.rot_angle = opts['rot_angle']
        self.netSize_l = opts['netSize_1_l']
        self.numb_aug_imgs = opts['numb_aug_imgs']
        self.border_treat_mode = opts['border_treat_mode']
        self.labCropRects_l = [1, 1, self.netSize_l[0], self.netSize_l[0]]
        self.sizeAugmentLabels = [self.netSize_l[0], self.netSize_l[0], 1, self.numb_aug_imgs]

        self.goodAugDatInd_l = []
        self.augData_c = []
        self.augLabels_c = []
        self.augClasses_c = []
        self.augWeights_c = []
        self.tmpAugDiffDatas_c = []

        self.ai_i = 1

    def __call__(self, sData_c, sLabels_c, sClasses_c, origWeights_c, sfD_c):
        sData_c = np.transpose(np.array(sData_c), (2, 1, 0)) #Getting the tensor to the matlab shape
        sLabels_c = np.transpose(np.array(sLabels_c), (2, 1, 0)) #Getting the tensor to the matlab shape

        transforms.functional.affine(sData_c, angle=0, translate=(254, 254), scale=(0, 0), shear=(0, 0))

        self.sizeAugData_l = [self.netSize_l[0], self.netSize_l[0], sData_c.shape[2], self.numb_aug_imgs]

        #Target Image Grid
        xGr, yGr = np.meshgrid(range(1, self.sizeAugData_l[0] + 1), range(1, self.sizeAugData_l[1] + 1), indexing='ij') #target Image grid

        #Sample from
        if self.has_smpl_dstrb:
            if sfD_c.size != 0:
                sfD = sfD_c
            else:
                raise ValueError('sfD_c is not found')

        #Cent for target image
        tMat = np.array([[1, 0, -1 * self.sizeAugData_l[0] / 2], [0, 1, -1 * self.sizeAugData_l[1] / 2], [0, 0, 1]]) #cent for target image

        #Random rotation
        phi = (2 * np.random.rand() - 1) * self.rot_angle
        tMat = np.dot(np.array([[math.cos(-phi), -math.sin(-phi), 0], [math.sin(-phi), math.cos(-phi), 0], [0, 0, 1]]), tMat)

        #Random scale
        lwrd = self.scale[0]
        upb = self.scale[1]
        s1 = (lwrd + (upb - lwrd) * np.random.rand())
        s2 = s1
        tMat = np.dot(np.array([[1 / s1, 0, 0], [0, 1 / s2, 0], [0, 0, 1]]), tMat)

        #Random shift
        shiftX = (2 * np.random.rand() - 1) * self.offset_magnitude[0]
        shiftY = (2 * np.random.rand() - 1) * self.offset_magnitude[1]
        tMat = np.dot(np.array([[1, 0, -shiftX], [0, 1, -shiftY], [0, 0, 1]]), tMat)
        
        #Cent for source image
        tMat = np.dot(np.array([[1, 0, sData_c.shape[0] / 2], [0, 1, sData_c.shape[1] / 2], [0, 0, 1]]), tMat)

        #Random deformation
        uDsplcms = np.random.randn(self.deform_numb_grid_pnts, self.deform_numb_grid_pnts)
        vDsplcms = np.random.randn(self.deform_numb_grid_pnts, self.deform_numb_grid_pnts)
        uDsplcms = np.ones(xGr.shape) * uDsplcms
        vDsplcms = np.ones(xGr.shape) * vDsplcms

        #Put all together
        xT = tMat[0, 0] * xGr + tMat[0, 1] * yGr + tMat[0, 2] + self.deform_magnitude * uDsplcms
        yT = tMat[1, 0] * xGr + tMat[1, 1] * yGr + tMat[1, 2] + self.deform_magnitude * vDsplcms

        #Random flipping
        if self.has_flip:
            rand_flip = random.randint(0, 3)
            if rand_flip == 0:
                xT = np.fliplr(xT)
                yT = np.fliplr(yT)
            elif rand_flip == 1:
                xT = np.flipud(xT)
                yT = np.flipud(yT)

        #Border treatment with mirroring, where necessary
        xT_copy = xT.copy()
        yT_copy = yT.copy()
        xT[xT < 1] = 2 - xT[xT < 1]
        yT[yT < 1] = 2 - yT[yT < 1]
        xT[xT > sData_c.shape[0]] = 2 * sData_c.shape[0] - xT[xT > sData_c.shape[0]]
        yT[yT > sData_c.shape[1]] = 2 * sData_c.shape[1] - yT[yT > sData_c.shape[1]]

        #sampling (by rejection)
        if self.has_smpl_dstrb:
            #An OK approximation of the matlab 'sfD = interp2d(xT, yT, sfD, 'linear')' interp2d method
            #ip = RectBivariateSpline(range(520), range(696), sfD)
            #sfD = ip.ev(xT, yT)
            sfD = self.interpn_python(sfD, xT, yT, method='linear')
            midP_l = [sfD.shape[0] / 2, sfD.shape[1] / 2]

            #Needs trials logic here?

        #Apply transformations
        #Apply to data
        augD_tmp = self.applyOpChannelWise(self.interpn_python, sData_c, self.sizeAugData_l[:3], xT, yT, method='bicubic')

        #Apply to labels
        tmp_labA = self.applyOpChannelWise(self.interpn_python, sLabels_c, [self.sizeAugData_l[0], self.sizeAugData_l[1], sLabels_c.shape[2]], xT, yT, method='nearest')
        #apply to weights
        if np.any(origWeights_c > 0):
            tmp_Weights = self.interpn_python(origWeights_c, xT, yT)
            #tmp_Weights[tmp_Weights < 0] = 0
            #tmp_Weights[tmp_Weights > 1] = 1
        else:
            tmp_Weights = np.zeros(tmp_labA[:, :, 0].shape)
        
        #apply to diff data
        ###Skipping parts concerning diffdata since they are not tested or functional in the matlab code

        #treat borders with mirror / zeros
        if self.border_treat_mode != 0:
            #treat data
            augD_tmp = self.treatBorders(augD_tmp, [sData_c.shape[0], sData_c.shape[1]], xT_copy, yT_copy, 0)
            #treat labels
            tmp_labA = self.treatBorders(tmp_labA, [sData_c.shape[0], sData_c.shape[1]], xT_copy, yT_copy, 0)
            #treat weights
            tmp_Weights = self.treatBorders(tmp_Weights, [sData_c.shape[0], sData_c.shape[1]], xT_copy, yT_copy, 0)
        
        # crop to the specific size
        #crop labels
        tmp_labA = self.applyOpChannelWise2(self.myImcrop, tmp_labA, [self.sizeAugmentLabels[0], self.sizeAugmentLabels[1], sLabels_c.shape[2]], self.labCropRects_l)
        #crop weights
        tmp_Weights = self.myImcrop(tmp_Weights, self.labCropRects_l)
        
        self.goodAugDatInd_l.append(self.ai_i)
    
        augData_c = augD_tmp
        augLabels_c = tmp_labA
        augWeights_c = tmp_Weights
        augClasses_c = sClasses_c
        sample = {'augData_c': np.array(augData_c, dtype=float), 'augLabels_c': np.array(augLabels_c, dtype=int), 'augClasses_c': np.array(augClasses_c, dtype=int), 'augWeights_c': np.array(augWeights_c, dtype=float)}
        return sample
    

    #Helper functions
    @staticmethod
    def interpn_python(x, xT, yT, method='linear'):
        if method == 'bicubic':
            ip = RectBivariateSpline(range(x.shape[0]), range(x.shape[1]), x)
            x = ip.ev(xT, yT)
        if method == 'linear':
            ip = RectBivariateSpline(range(x.shape[0]), range(x.shape[1]), x, kx=1, ky=1)
            x = ip.ev(xT, yT)
        elif method == 'nearest':
            x = map_coordinates(x, [xT.ravel(), yT.ravel()], order=3, mode='nearest').reshape(xT.shape)
        return x

    @staticmethod
    def applyOpChannelWise(func, A, outShape, xT, yT, method='linear'):
        B = np.zeros(outShape, dtype=A.dtype)
        for ch_i in range(0, A.shape[2]):
            B[:, :, ch_i] = func(A[:, :, ch_i], xT, yT, method=method)
        return B
    
    @staticmethod
    def applyOpChannelWise2(func, A, outShape, labCropRects_l):
        B = np.zeros(outShape, dtype=A.dtype)
        for ch_i in range(0, A.shape[2]):
            B[:, :, ch_i] = func(A[:, :, ch_i], labCropRects_l)
        return B

    @staticmethod
    def myImcrop(im, labCropRects_l):
        first_part = len(np.arange(labCropRects_l[0] - 1, labCropRects_l[2]))
        second_part = len(np.arange(labCropRects_l[1] - 1, labCropRects_l[3]))
        if not ((first_part == im.shape[0]) and (second_part == im.shape[1])):
            im = im[labCropRects_l[0] - 1:labCropRects_l[2], labCropRects_l[1] - 1:labCropRects_l[3]]
        return im

    @staticmethod
    def treatBorders(sData_c, origDataSize, xT, yT, mode):
        if len(sData_c.shape) == 2:
            sData_c = sData_c[:, :, np.newaxis]
        rmap = lambda x : np.tile(x[:, :, np.newaxis], [1, 1, sData_c.shape[2]])
        if mode == 0:
            sData_c[rmap(xT < 1)] = 0
            sData_c[rmap(xT > origDataSize[0])] = 0
            sData_c[rmap(yT < 1)] = 0
            sData_c[rmap(yT > origDataSize[1])] = 0
        elif mode == 1:
            sData_c[rmap(xT < 1)] = 1
            sData_c[rmap(xT > origDataSize[0])] = 1
            sData_c[rmap(yT < 1)] = 1
            sData_c[rmap(yT > origDataSize[1])] = 1
        return sData_c