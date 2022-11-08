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
from precomputeSFD_f import precomputeSFD_f
from ignoreMasksOnBorders_f import ignoreMasksOnBorders_f
from genDsksOfadaptSize_f import genDsksOfadaptSize_f
from weightSpaceBtwDsks_f import weightSpaceBtwDsks_f
from cropBlob_f import cropBlob_f
from prepareData_f import prepareData_f
from matplotlib import pyplot as plt
import scipy.io as sio
from tqdm import tqdm
import multiprocessing

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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
                self.img_labels.append(torch.from_numpy(f['labels'][()])) #Get the values of the hdf5 labels as array
                self.images_c.append(torch.from_numpy(f['data'][()]))

                labels_c.append(np.transpose(f['labels'][()], (2, 1, 0))) #Get the labels of the hdf5 dataset as array
                weights_c.append((np.transpose(f['ignore'][()][0, :, :]) == 0).astype('float32')) #Load the weights with the same form as in matlab
                classes_c.append(np.array(f['classes'][()]))
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
        #self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #image = self.batchIms_c[idx]
        #label = self.img_labels[idx]
        sample = {'batchIms_c': self.batchIms_c[idx], 'batchLabs_c': self.batchLabs_c[idx], 
        'batchCls_c': self.batchCls_c, 'batchWghts_c': self.batchWghts_c[idx], 'img_name': self.img_names[idx]}
        if self.transform:
            sample = self.transform(self.batchIms_c[idx], self.batchLabs_c[idx], self.batchCls_c[idx], self.batchWghts_c[idx], self.batchSfD_c[idx])
            sample['img_name'] =  self.img_names[idx]
        #if self.target_transform:
            #label = self.target_transform(label)
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
        sData_c = torch.permute(sData_c, (2, 1, 0)) #Getting the tensor to the matlab shape
        sLabels_c = torch.permute(sLabels_c, (2, 1, 0)) #Getting the tensor to the matlab shape

        #Matlab reproducibility test
        '''
        matvars = sio.loadmat('aug.mat')
        sData_c = matvars['sData_c'][0][0]
        sLabels_c = matvars['sLabels_c'][0][0]
        sClasses_c = matvars['sClasses_c'][0][0]
        origWeights_c = matvars['origWeights_c'][0][0]
        '''

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
            ip = RectBivariateSpline(range(520), range(696), sfD)
            sfD = ip.ev(xT, yT)
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
        
        """
        if all(tmp_Weights(:) == 0)
            warning('all(Ignore) is true => skipping')
            continue;
        end
        """
        self.goodAugDatInd_l.append(self.ai_i)
        """
        self.augData_c.append(augD_tmp)
        self.augLabels_c.append(tmp_labA)
        self.augWeights_c.append(tmp_Weights)
        #self.augClasses_c.append(sClasses_c)
        """

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
        B = np.zeros(outShape, dtype=type(A))
        for ch_i in range(0, A.shape[2]):
            B[:, :, ch_i] = func(A[:, :, ch_i], xT, yT, method=method)
        return B
    
    @staticmethod
    def applyOpChannelWise2(func, A, outShape, labCropRects_l):
        B = np.zeros(outShape, dtype=type(A))
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

'''
prepareData_f.m
'''
"""
data_dir = '../DiskMask/data_specs/c_elegans/example_data/ds_tr/'
data_files = os.listdir(data_dir)

batchIms_c = []
for data_file in data_files:
    with h5py.File(data_dir + data_file, "r") as f:
        batchIms_c.append(f['data'][()]) #Get the values of the hdf5 dataset as array

NUMB_AUG_IMGS = 10
numb_data = len(batchIms_c)

#Temporary opts for augmentation
augment_opts = {'has_smpl_dstrb': 1, 'has_flip': 1, 'deform_magnitude': 0, 
                'deform_numb_grid_pnts': 1, 'offset_magnitude': [194, 282], 
                'rot_angle': 6.283185307179586, 'scale': [1, 1], 'netSize_1_l': [508, 324],'numb_aug_imgs': 1,
                'border_treat_mode': 1, 'disk_radius': 9, 'rp_id': 0}

data_transform = AugmentData_v2_f(opts=augment_opts)
elegans_dataset = ElegansDataset(data_dir=data_dir, transform=data_transform)
train_dataloader = DataLoader(elegans_dataset, batch_size=1, sampler=torch.utils.data.RandomSampler(elegans_dataset, replacement=True))

augData_c = []
augLabs_c = []
augWeights_c = []

tmp_augLabs_c = []
tmp_augClasses_c = []
tmp_augWeights_c = []
for aim_i in tqdm(range(NUMB_AUG_IMGS)):
    #augment data, data will be croped to the net input
    augData, tmp_augL, tmp_augC, augWeight, img_name = next(iter(train_dataloader)).values()

    #Matlab reproducibility test
    '''
    repro = sio.loadmat('preparedata.mat')
    augData = repro['A'][0][0]
    tmp_augL = repro['B'][0][0]
    tmp_augC = repro['C'][0][0]
    augWeight = repro['D'][0][0]
    '''

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
    kelko = sio.loadmat('mat_vars3')
    tmp_test_output = sio.loadmat('gendiskoutput.mat')
    dskswCl, rpIDs, rpWgtsInsDsks = genDsksOfadaptSize_f(tmp_augLabs_c[aim_i], tmp_augClasses_c[aim_i], augment_opts['disk_radius'], augment_opts['rp_id'])
    #dskswCl, rpIDs, rpWgtsInsDsks = genDsksOfadaptSize_f(kelko['kelko1'], kelko['kelko2'], augment_opts['disk_radius'], augment_opts['rp_id'])
    #dskswCl2 = tmp_test_output['dskswCl']
    #rpIDs2 = tmp_test_output['rpIDs']
    #rpWgtsInsDsks2 = tmp_test_output['rpWgtsInsDsks']
    
    #weight space btw. object disks
    rpWgtsBtwDsks = weightSpaceBtwDsks_f(dskswCl, elegans_dataset.sigma_px, elegans_dataset.w_0)
    #rpWgtsBtwDsks = weightSpaceBtwDsks_f(tmp_test_output['dskswCl'], elegans_dataset.sigma_px, elegans_dataset.w_0)
    #rpWgtsBtwDsks2 = sio.loadmat('rpWgtsBtwDsks2.mat')['rpWgtsBtwDsks']
    rpWgtsBtwDsks[dskswCl > 0] = 0 #set to zero at object disks

    #putting weights together
    ignoreMask = np.logical_or(np.logical_or(tmp_augWeights_c[aim_i] == 0, dskswCl == 255), np.array(ignoreOB, dtype=bool))
    wghtCls = np.ones(dskswCl.shape)
    wghtCls = wghtCls + rpWgtsInsDsks + rpWgtsBtwDsks
    wghtCls[ignoreMask] = 0

    # crop sLabels_c and sWeights_c
    #dskswCl_matlab = sio.loadmat('cropblob.mat')['dskswCl']
    #wghtCls_matlab = sio.loadmat('cropblob.mat')['wghtCls']
    #rpIDs_matlab = sio.loadmat('cropblob.mat')['rpIDs']
    #tmp_augLabs_c_matlab = sio.loadmat('cropblob.mat')['tmp_augLabs_c']
    augLabs_c1 = cropBlob_f(dskswCl, augment_opts['netSize_1_l'])
    augWeights_c.append(np.squeeze(cropBlob_f(wghtCls, augment_opts['netSize_1_l'])))
    augLabs_c2 = cropBlob_f(rpIDs, augment_opts['netSize_1_l'])
    augLabs_c3 = cropBlob_f(tmp_augLabs_c[aim_i], augment_opts['netSize_1_l'])
    augLabs_c.append([np.squeeze(augLabs_c1), np.squeeze(augLabs_c2), np.squeeze(augLabs_c3)])

print('end of prepare_data.f')
"""
'''
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img.permute(1, 2, 0))
plt.show()
print(f"Label: {label}")
'''