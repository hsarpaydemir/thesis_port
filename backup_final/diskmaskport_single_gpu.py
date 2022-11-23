from skimage.measure import label
import os
import torch
from torch import nn, relu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import h5py
import multiprocessing
from helpers.loadData import loadData
from helpers.detectRPsOnBorders_f import detectRPsOnBorders_f
from helpers.activateRPandMask_v2_f import activateRPandMask_v2_f
from helpers.cropBlob_f import cropBlob_f
import random
from skimage.morphology import thin
import pdb
import matplotlib.pyplot as plt
import time
from helpers.datasetUtility import AugmentData_v2_f, ElegansDataset
from torchvision import transforms
import scipy.io as sio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'Using {device} device')

class CombinedTransforms2(object):
    def __init__(self):
        pass

    def __call__(self, image, labels, weights):
        rand_number = random.randint(0, 3)
        translate_x = random.randint(-192, 192)
        translate_y = random.randint(-242, 242)
        rotation_angle = random.randint(-180, 180)
        crop_size = (508, 508)

        image = torch.from_numpy(image)
        labels = torch.from_numpy(labels.astype(int))
        weights = torch.from_numpy(weights)
        image = torch.permute(image, (0, 2, 1))
        labels = torch.permute(labels, (0, 2, 1))
        weights = weights.unsqueeze(0)

        img_tmp = MyRandomFlip(random_number=rand_number)(image)
        #img_tmp = BorderTreatment()(img_tmp)
        img_tmp = transforms.functional.affine(img_tmp, rotation_angle, (translate_x, translate_y), 1, 0)
        img_tmp = transforms.functional.center_crop(img_tmp, crop_size)

        labels_tmp = MyRandomFlip(random_number=rand_number)(labels)
        #labels_tmp = BorderTreatment()(labels_tmp)
        labels_tmp = transforms.functional.affine(labels_tmp, rotation_angle, (translate_x, translate_y), 1, 0)
        labels_tmp = transforms.functional.center_crop(labels_tmp, crop_size)

        weights_tmp = MyRandomFlip(random_number=rand_number)(weights)
        #weights_tmp = BorderTreatment()(weights_tmp)
        weights_tmp = transforms.functional.affine(weights_tmp, rotation_angle, (translate_x, translate_y), 1, 0)
        weights_tmp = transforms.functional.center_crop(weights_tmp, crop_size)
        
        return img_tmp.cpu().detach().numpy(), labels_tmp.cpu().detach().numpy(), weights_tmp.cpu().detach().numpy()

class CombinedTransforms(object):
    def __init__(self):
        pass

    def __call__(self, image, labels, weights):
        rand_number = random.randint(0, 3)
        translate_x = random.randint(-192, 192)
        translate_y = random.randint(-242, 242)
        rotation_angle = random.randint(-180, 180)
        crop_size = (508, 508)

        image = transforms.ToTensor()(image)
        labels = transforms.ToTensor()(labels.astype(int))
        weights = transforms.ToTensor()(weights)
        image = torch.permute(image, (1, 0, 2))
        labels = torch.permute(labels, (1, 0, 2))

        img_tmp = MyRandomFlip(random_number=rand_number)(image)
        img_tmp = BorderTreatment()(img_tmp)
        img_tmp = transforms.functional.affine(img_tmp, rotation_angle, (translate_x, translate_y), 1, 0)
        img_tmp = transforms.functional.center_crop(img_tmp, crop_size)

        labels_tmp = MyRandomFlip(random_number=rand_number)(labels)
        labels_tmp = BorderTreatment()(labels_tmp)
        labels_tmp = transforms.functional.affine(labels_tmp, rotation_angle, (translate_x, translate_y), 1, 0)
        labels_tmp = transforms.functional.center_crop(labels_tmp, crop_size)

        weights_tmp = MyRandomFlip(random_number=rand_number)(weights)
        weights_tmp = BorderTreatment()(weights_tmp)
        weights_tmp = transforms.functional.affine(weights_tmp, rotation_angle, (translate_x, translate_y), 1, 0)
        weights_tmp = transforms.functional.center_crop(weights_tmp, crop_size)
        
        return img_tmp.cpu().detach().numpy(), labels_tmp.cpu().detach().numpy(), weights_tmp.cpu().detach().numpy()

class BorderTreatment(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        #Random flipping
        sample_original = sample
        sample = torch.permute(sample, (2, 1, 0))
        out_tensor = torch.zeros(sample.shape[2], sample.shape[1] * 2, sample.shape[0] * 2)
        out_tensor[:, :sample.shape[1], :sample.shape[0]] = sample_original
        out_tensor[:, :sample.shape[1], sample.shape[0]:] = transforms.functional.hflip(sample_original)
        out_tensor[:, sample.shape[1]:, :sample.shape[0]] = transforms.functional.vflip(sample_original)
        out_tensor[:, sample.shape[1]:, sample.shape[0]:] = transforms.functional.vflip(transforms.functional.hflip(sample_original))
        #return torch.permute(out_tensor, (2, 1, 0))
        return out_tensor

class MyRandomFlip(object):
    def __init__(self, random_number):
        self.random_number = random_number

    def __call__(self, sample):
        #Random flipping
        if self.random_number == 0:
            sample = transforms.functional.hflip(sample)
        elif self.random_number == 1:
            sample = transforms.functional.vflip(sample)
        return sample   

'''
class MyRandomFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        #Random flipping
        rand_flip = random.randint(0, 3)
        if rand_flip == 0:
            sample = transforms.functional.hflip(sample)
        elif rand_flip == 1:
            sample = transforms.functional.vflip(sample)
        return sample   
'''

class CenterForTarget(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        out_sample = torch.zeros(int(self.output_size[0] / 2) + sample.shape[2], int(self.output_size[1] / 2) + sample.shape[1], 3)
        sample = torch.permute(sample, (2, 1, 0))
        #out_sample[int(self.output_size[0] / 2):, int(self.output_size[1] / 2):, :] = sample[:int(self.output_size[0] / 2), :int(self.output_size[1] / 2), :]
        out_sample[int(self.output_size[0] / 2):, int(self.output_size[1] / 2):, :] = sample

        return torch.permute(out_sample, (2, 1, 0))

class MyCenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        crop_output = transforms.functional.center_crop(sample, output_size=(self.output_size[0], self.output_size[1]))

        return crop_output

class DiskMaskNet(nn.Module):
    def __init__(self):
        super(DiskMaskNet, self).__init__()
        self.flatten = nn.Flatten()
        self.n1_d1c = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=2, padding=0), 
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0), 
            nn.ReLU(),
        ) 

        self.n1_d2c = nn.Sequential( 
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.n1_d3c = nn.Sequential( 
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        ) 
        
        self.n1_d4c = nn.Sequential( 
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        ) 

        self.n1_u3d = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        ) 

        self.n1_u2d = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        ) 

        self.n1_u1d = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        ) 

        self.predict_ref_points = nn.Sequential(
            #######Predict ref points
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding=0),
            ####SoftmaxWithLoss here
        )

        self.predict_instance_masks = nn.Sequential(
            #######Predict instance masks
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        self.n2_d0b = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=2, padding=0),
            nn.ReLU(),
        )

        self.n2_d1c = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.pooling_deconet = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.n2_d2c = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )
        
        self.n2_d3c = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.n2_d4c = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.n2_u3a = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
        )

        self.n2_u3d = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.n2_u2a = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
        )

        self.n2_u2d = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.n2_u1a = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
        )

        self.n2_u1d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.n2_u0d = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
        )

        self.pred_semseg = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0)
            #SigmoidCrossEntropyLoss
        )

    def forward(self, x, gates):
        #x = self.flatten(x)
        #logits = self.first_part(x)
        #return logits

        #Predict ref points
        
        #EncoNet
        #Encoder Part
        n1_d1c = self.n1_d1c(x) 
        n1_d2c = self.n1_d2c(n1_d1c)
        n1_d3c = self.n1_d3c(n1_d2c)
        n1_d4c = self.n1_d4c(n1_d3c)

        #Decoder Part
        n1_u3d = self.n1_u3d(n1_d4c)
        n1_u2d = self.n1_u2d(n1_u3d)
        n1_u1d = self.n1_u1d(n1_u2d)
        
        pred_cls = self.predict_ref_points(n1_u1d) #(1, 2, 324, 324)

        #predict instance masks
        obj_rp = self.predict_instance_masks(n1_u1d) #(1, 64, 324, 324)

        
        #Activate some ref points
        sem_scores_tld = torch.tile(gates, (1, 64, 1, 1)) #####Axis = 1 for caffe tile operation, (1, 64, 324, 324)
        obj_rp_act = torch.mul(sem_scores_tld, obj_rp) #(1, 64, 324, 324)

        #### crop 324 to in = 316 to get out = 132
        obj_rp_cr = obj_rp_act[:, :, 4:4 + 316, 4:4 + 316] #Crop layer, (1, 64, 316, 316)

        #DecoNet
        #Encoder Part
        n2_d0b = self.n2_d0b(obj_rp_cr) #(1, 64, 156, 156)
        #n2_d0b_corrected = self.correction_layer(n2_d0b) #Corrected n2_d0b to fit into concat_n2_d2a_u1d layer
        n1_d1c_cropped = n1_d1c[:, :, 46:46 + n2_d0b.shape[2], 46:46 + n2_d0b.shape[3]]
        n2_d2a_u1d = torch.cat((n2_d0b, n1_d1c_cropped), dim=1)
        n2_d1c = self.n2_d1c(n2_d2a_u1d)

        n2_d2a = self.pooling_deconet(n2_d1c)
        n1_d2c_cropped = n1_d2c[:, :, 22:22 + n2_d2a.shape[2], 22:22 + n2_d2a.shape[3]]
        n2_d2a_u2d = torch.cat((n2_d2a, n1_d2c_cropped), dim=1)
        n2_d2c = self.n2_d2c(n2_d2a_u2d)
        
        n2_d3a = self.pooling_deconet(n2_d2c)
        n1_d3c_cropped = n1_d3c[:, :, 10:10 + n2_d3a.shape[2], 10:10 + n2_d3a.shape[3]]
        n2_d3a_u3d = torch.cat((n2_d3a, n1_d3c_cropped), dim=1)
        n2_d3c = self.n2_d3c(n2_d3a_u3d)

        n2_d4a = self.pooling_deconet(n2_d3c)
        n1_d4c_cropped = n1_d4c[:, :, 4:4 + n2_d4a.shape[2], 4:4 + n2_d4a.shape[3]]
        n2_d3a_d4c = torch.cat((n2_d4a, n1_d4c_cropped), dim=1)
        n2_d4c = self.n2_d4c(n2_d3a_d4c)

        #Decoder Part
        n2_u3a = self.n2_u3a(n2_d4c)
        n2_d3c_cropped = n2_d3c[:, :, 4:4 + n2_u3a.shape[2], 4:4 + n2_u3a.shape[3]]
        n2_u3b = torch.cat((n2_u3a, n2_d3c_cropped), dim=1)
        n2_u3d = self.n2_u3d(n2_u3b)

        n2_u2a = self.n2_u2a(n2_u3d)
        n2_d2c_cropped = n2_d2c[:, :, 16:16 + n2_u2a.shape[2], 16:16 + n2_u2a.shape[3]]
        n2_u2b = torch.cat((n2_u2a, n2_d2c_cropped), dim=1)
        n2_u2d = self.n2_u2d(n2_u2b)

        n2_u1a = self.n2_u1a(n2_u2d)
        n2_d1c_cropped = n2_d1c[:, :, 40:40 + n2_u1a.shape[2], 40:40 + n2_u1a.shape[3]]
        n2_u1b = torch.cat((n2_u1a, n2_d1c_cropped), dim=1)
        n2_u1d = self.n2_u1d(n2_u1b)

        n2_u0d = self.n2_u0d(n2_u1d)

        #Semseg
        pred_semseg = self.pred_semseg(n2_u0d)
        return pred_cls, pred_semseg

#MSRA weight initialization in caffe file
def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

def main():
    #Initialize model, optimizer and criterion
    model = DiskMaskNet().to(device)
    #model.apply(weights_init)
    model.load_state_dict(torch.load('/home/haydemir/Desktop/connectivity/thesis/pythorch_port/fixed_tensor_model_zerograd_15000.pt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300000, gamma=0.1) #Reduce the lr to 0.00001 from 0.0001 after iteration 300k
    enco_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none') #Same with caffe SoftmaxWithLoss layer (Softmax followed by multinomial logistic loss)
    deco_criterion = nn.BCEWithLogitsLoss(reduction='none') #Same with caffe SigmoidCrossEntropyLoss  layer (Combines sigmoid layer and the BCELoss)
    #deco_criterion = nn.CrossEntropyLoss(ignore_index=255)

    augment_opts = {'has_smpl_dstrb': 1, 'has_flip': 1, 'deform_magnitude': 0, 
                        'deform_numb_grid_pnts': 1, 'offset_magnitude': [194, 282], 
                        'rot_angle': 6.283185307179586, 'scale': [1, 1], 'netSize_1_l': [508, 324],'numb_aug_imgs': 1,
                        'border_treat_mode': 1, 'disk_radius': 9, 'rp_id': 0, 'has_objcts_in_ignr_reg': False,
                        'keep_prob': 0.5, 'netSize_2_l': [316, 132]}

    '''
    aug_transforms = transforms.Compose([
        MyRandomFlip(),
        BorderTreatment(),
        transforms.RandomAffine(degrees=180, translate=(192 / 508, 242 / 508)),
        transforms.CenterCrop(508),
    ])
    '''

    data_dir = '../DiskMask/data_specs/c_elegans/example_data/ds_tr/'
    data_transform = AugmentData_v2_f(opts=augment_opts)
    elegans_dataset = ElegansDataset(data_dir=data_dir, transform=CombinedTransforms2())

    for epoch in tqdm(range(4000)):  
        running_loss = 0.0
        worker_id, augData_c, augLabs_c, augWeights_c = loadData(0, elegans_dataset, augment_opts)

        for r_d in range(len(augData_c)):
            im = augData_c[r_d]
            if im.shape[2] == 1:
                im = np.tile(im, (1, 1, 3))

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
            label_2D_cls = torch.tensor(dskswCl).to(device, dtype=torch.long)
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
            #augLabs, bwDskswCl = activateRPandMask_v2_f(rpIDs, augLabs, ccDsks, bwDskswCl, ref2reject, 1)

            #augment gateway size
            bwDskswCl = thin(bwDskswCl, random.randint(1, augment_opts['disk_radius'] - 1) - 1)
            #bwDskswCl = thin(bwDskswCl, 1)

            #Need to connect the input of enconet('data' in caffe) to im here
            #Need to connect the input of enconet('gates') to bwDskswCl here
            im = torch.unsqueeze(torch.permute(im, (2, 0, 1)), 0).to(device, dtype=torch.float)
            bwDskswCl = torch.tensor(bwDskswCl).to(device)
            pred_cls, pred_semseg = model(im, bwDskswCl)

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
            augLabs[ignore.astype(int) == 1] = 255

            #Need to connect the input of enconet('gt_semseg') to augLabs here
            gt_semseg = torch.tensor(augLabs).to(device)

            #track loss
            
            #Forward pass
            #pdb.set_trace()
            #sigmoid = nn.Sigmoid()
            enco_loss = enco_criterion(pred_cls, label_2D_cls.unsqueeze(0)) #Loss in the EncoNet part
            #Need to connect the input of enconet('weight_2D_cls') to wghtCls here
            enco_loss = (enco_loss * torch.tensor(wghtCls).to(device).unsqueeze(0)).mean() #Weighting the losses with augmented weights
            deco_loss = deco_criterion(pred_semseg, gt_semseg.unsqueeze(0).unsqueeze(0)) #Loss in the DecoNet part
            deco_loss[gt_semseg.unsqueeze(0).unsqueeze(0) == 255] = 0
            deco_loss = deco_loss.sum() / (gt_semseg.unsqueeze(0).unsqueeze(0) != 255).sum() #Getting rid of the losses with ignore mask (gt_semseg == 255)
            loss = enco_loss + deco_loss #Losses are both weighted 1 in the caffe file, thus they are added for the backward pass
            running_loss += loss.item()
            print('Enco loss: ', enco_loss.item())
            print('Deco loss: ', deco_loss.item())
            print('Total loss: ', loss.item())
            #Backward and optimize
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if epoch % 10 == 0 :
            print("Runnig_Loss: {}".format(running_loss))

    torch.save(model.state_dict(), '/home/haydemir/Desktop/connectivity/thesis/pythorch_port/fixed_tensor_model_zerograd_continued_from14000.pt')

if __name__ == '__main__':
    main()