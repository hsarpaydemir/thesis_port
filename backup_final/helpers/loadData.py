from helpers.datasetUtility import AugmentData_v2_f, ElegansDataset, MyCompose
from torch.utils.data import DataLoader
import torch
from helpers.prepareData_f import prepareData_f
import time

def loadData(worker_id, elegans_dataset, augment_opts):
    #Temporary opts for augmentation
    train_dataloader = DataLoader(elegans_dataset, batch_size=1, sampler=torch.utils.data.RandomSampler(elegans_dataset, replacement=True))

    augData_c, augLabs_c, augWeights_c = prepareData_f(train_dataloader, augment_opts)

    return (worker_id, augData_c, augLabs_c, augWeights_c)