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
from scipy.interpolate import interpn, interp2d, LinearNDInterpolator, RectBivariateSpline, SmoothBivariateSpline, LSQBivariateSpline
from scipy.ndimage import map_coordinates
from helpers.precomputeSFD_f import precomputeSFD_f

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
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        data_files = os.listdir(data_dir)
        self.img_labels = []
        self.images_c= []

        labels_c = []
        weights_c = []

        for data_file in data_files:
            with h5py.File(data_dir + data_file, "r") as f:
                self.img_labels.append(torch.from_numpy(f['labels'][()])) #Get the values of the hdf5 labels as array
                self.images_c.append(torch.from_numpy(f['data'][()]))

                labels_c.append(np.transpose(f['labels'][()], (2, 1, 0))) #Get the labels of the hdf5 dataset as array
                weights_c.append((np.transpose(f['ignore'][()][0, :, :]) == 0).astype('float32')) #Load the weights with the same form as in matlab

        #Randomly permuting the images and respective arrays
        batchSfD_c = precomputeSFD_f(labels_c, weights_c)
        batchIms_c = self.images_c 
        batchLabs_c = self.img_labels
        permute_temp = list(zip(batchIms_c, batchLabs_c, batchSfD_c))
        random.shuffle(permute_temp)
        self.batchIms_c, self.batchLabs_c, self.batchSfD_c = zip(*permute_temp)

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.batchIms_c[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(self.batchIms_c[idx], self.batchLabs_c[idx], self.batchSfD_c[idx])
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

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
        self.labCropRects_l = [1, 1, self.netSize_l[0], self.netSize_l[0]]
        self.sizeAugmentLabels = [self.netSize_l[0], self.netSize_l[0], 1, self.numb_aug_imgs]

    def __call__(self, sData_c, sLabels_c, sfD_c):
        sData_c = torch.permute(sData_c, (2, 1, 0)) #Getting the tensor to the matlab shape
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
        xT_copy = xT
        yT_copy = yT
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
        augD_tmp = self.applyOpChannelWise(self.interpn_python, sData_c, self.sizeAugData_l[:3], xT, yT)

        tmp_labA = self.applyOpChannelWise(self.interpn_python, sLabels_c, [self.sizeAugData_l.shape[0], self.sizeAugData_l[1], sLabels_c.shape[2]], xT, yT)

        return sData_c
    

    #Helper functions
    @staticmethod
    def interpn_python(x, xT, yT):
        ip = RectBivariateSpline(range(x.shape[0]), range(x.shape[1]), x)
        x = ip.ev(xT, yT)
        return x

    @staticmethod
    def applyOpChannelWise(func, A, outShape, xT, yT):
        B = np.zeros(outShape, dtype=type(A))
        for ch_i in range(0, A.shape[2]):
            B[:, :, ch_i] = func(A[:, :, ch_i], xT, yT)
        return B

    @staticmethod
    def myImcrop(im, labCropRects_l):
        first_part = len(np.arange(labCropRects_l[0], labCropRects_l[2] + 1))
        second_part = len(np.arange(labCropRects_l[1], labCropRects_l[3] + 1))
        if not ((first_part == im.shape[0]) and (second_part == im.shape[1])):
            return im[labCropRects_l[0]:labCropRects_l[2], labCropRects_l[1]:labCropRects_l[3], 0]
        else:
            return im

    #def treatBorders(sData_c, origDataSize, xT, yT, mode):
    

data_dir = '../DiskMask/data_specs/c_elegans/example_data/ds_tr/'
data_files = os.listdir(data_dir)

batchIms_c = []
for data_file in data_files:
    with h5py.File(data_dir + data_file, "r") as f:
        batchIms_c.append(f['data'][()]) #Get the values of the hdf5 dataset as array

NUMB_AUG_IMGS = 10
numb_data = len(batchIms_c)

for i in range(NUMB_AUG_IMGS):
    rnd_i = random.randrange(0, numb_data) #Get random index to choose random img from the batch

#Temporary opts for augmentation
augment_opts = {'has_smpl_dstrb': 1, 'has_flip': 1, 'deform_magnitude': 0, 
                'deform_numb_grid_pnts': 1, 'offset_magnitude': [194, 282], 
                'rot_angle': 6.283185307179586, 'scale': [1, 1], 'netSize_1_l': [508, 324],'numb_aug_imgs': 1}
data_transform = AugmentData_v2_f(opts=augment_opts)

elegans_dataset = ElegansDataset(data_dir=data_dir, transform=data_transform)
dataset_first_element = elegans_dataset[0]
train_dataloader = DataLoader(elegans_dataset, batch_size=64, shuffle=True)


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