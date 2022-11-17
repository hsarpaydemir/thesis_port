import torch
from diskmaskport_single_gpu import DiskMaskNet
from torch import nn
import h5py
import numpy as np
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import time

#MSRA weight initialization in caffe file
def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

model = DiskMaskNet()
model.apply(weights_init)
model.load_state_dict(torch.load('/home/haydemir/Desktop/connectivity/thesis/pythorch_port/second_model.pt'))
model.eval()

kelo = None
with h5py.File('/home/haydemir/Desktop/connectivity/thesis/DiskMask/data_specs/c_elegans/example_data/ds_te/C17_1_15_1_mod.h5', "r") as f:
    kelo = torch.from_numpy(f['data'][()])

transform = transforms.CenterCrop((508, 508))
kelo = transform(kelo).unsqueeze(0)
print(kelo.shape)

gates = torch.rand(1, 1, 324, 324)

pred_cls, pred_semseg = model(kelo, gates)

transform2 = transforms.CenterCrop((132, 132))

kelo2 = transform2(kelo)
kelo2[torch.tile(pred_semseg, (1, 3, 1, 1)) > 0] = 130

plt.imshow(np.squeeze(kelo2).permute(1, 2, 0))
plt.show()

utils.draw_segmentation_masks()

time.sleep(10)