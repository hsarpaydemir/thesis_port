from cProfile import label
import os
import torch
from torch import nn, relu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import h5py

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'Using {device} device')

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

    def forward(self, x):
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
        obj_rp_cr = obj_rp_act[:, :, 4:4 + DummyData_316.shape[2], 4:4 + DummyData_316.shape[3]] #Crop layer, (1, 64, 316, 316)

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
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300000, gamma=0.1) #Reduce the lr to 0.00001 from 0.0001 after iteration 300k
    enco_criterion = nn.CrossEntropyLoss() #Same with caffe SoftmaxWithLoss layer (Softmax followed by multinomial logistic loss)
    deco_criterion = nn.BCEWithLogitsLoss() #Same with caffe SigmoidCrossEntropyLoss  layer (Combines sigmoid layer and the BCELoss)
    
    #Declaring dummy data
    filename = "../DiskMask/data_specs/c_elegans/example_data/ds_tr/C17_1_15_1_mod.h5"
    f = h5py.File(filename, 'r')
    label_2D_cls = f["labels"]
    transform = transforms.CenterCrop((324, 324))
    label_2D_cls = transform(torch.tensor(label_2D_cls[0])).type(torch.LongTensor) ###Either class 0 or not, binary class this way (2 classes as in the paper). Cropped in the middle temporarily
    #label_2D_cls = torch.randint(2, (10, 324, 324), device=device) ###Mock data
    X = torch.rand((10, 3, 508, 508), device=device)
    
    gt_semseg = torch.rand((10, 1, 132, 132), device=device)
    global gates
    gates = torch.rand(1, 1, 324, 324).to(device=device)
    global DummyData_316
    DummyData_316 = torch.zeros(1, 64, 316, 316).to(device=device)

    for epoch in tqdm(range(600000)):
        for i in range(10):
            #Forward pass
            pred_cls, pred_semseg = model(torch.unsqueeze(X[i], 0))
            enco_loss = enco_criterion(pred_cls, torch.unsqueeze(label_2D_cls, 0)) #Loss in the EncoNet part
            deco_loss = deco_criterion(pred_semseg, torch.unsqueeze(gt_semseg[i], 0)) #Loss in the DecoNet part
            loss = enco_loss + deco_loss #Losses are both weighted 1 in the caffe file, thus they are added for the backward pass

            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print("Loss: {}".format(loss.item()))

    f.close()
if __name__ == '__main__':
    main()