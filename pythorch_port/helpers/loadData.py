from datasetUtility import AugmentData_v2_f, ElegansDataset, MyCompose
from torch.utils.data import DataLoader
import torch
from prepareData_f import prepareData_f

def loadData(worker_id):
    #Temporary opts for augmentation
    augment_opts = {'has_smpl_dstrb': 1, 'has_flip': 1, 'deform_magnitude': 0, 
                    'deform_numb_grid_pnts': 1, 'offset_magnitude': [194, 282], 
                    'rot_angle': 6.283185307179586, 'scale': [1, 1], 'netSize_1_l': [508, 324],'numb_aug_imgs': 1,
                    'border_treat_mode': 1, 'disk_radius': 9, 'rp_id': 0}

    data_dir = '../DiskMask/data_specs/c_elegans/example_data/ds_tr/'
    data_transform = AugmentData_v2_f(opts=augment_opts)
    elegans_dataset = ElegansDataset(data_dir=data_dir, transform=data_transform)
    train_dataloader = DataLoader(elegans_dataset, batch_size=1, sampler=torch.utils.data.RandomSampler(elegans_dataset, replacement=True))

    augData_c, augLabs_c, augWeights_c = prepareData_f(train_dataloader, augment_opts)
    return (worker_id, augData_c, augLabs_c, augWeights_c)