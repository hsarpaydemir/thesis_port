from cProfile import label
import os
import torch
from torch import nn, relu
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class DiskMaskNet(nn.Module):
    def __init__(self):
        super(DiskMaskNet, self).__init__()
        self.flatten = nn.Flatten()
        self.n1_d1c = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=2, padding=0), #(1, 64, 252, 252)
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0), #(1, 64, 250, 250)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0), #(1, 64, 248, 248)
            nn.ReLU(),
        )
        self.first_part = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=0),
            nn.ReLU(),

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

        self.correction_layer = nn.Sequential( ##Correction layer used to accoutn for the mismatch in the concat operation
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=2, padding=32, output_padding=1), 
            nn.ReLU(),
        )

        self.n2_d2a = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        

    def forward(self, x):
        #x = self.flatten(x)
        #logits = self.first_part(x)
        #return logits

        #Predict ref points
        
        n1_d1c = self.n1_d1c(x) #(1, 64, 248, 248)
        n1_u1d = self.first_part(n1_d1c) #(1, 256, 164, 164)
        pred_cls = self.predict_ref_points(n1_u1d)

        #predict instance masks
        obj_rp = self.predict_instance_masks(n1_u1d) #(1, 64, 324, 324)

        #Activate some ref points
        sem_scores_tld = torch.tile(gates, (1, 64, 1, 1)) #####Axis = 1 for caffe tile operation, (1, 64, 324, 324)
        obj_rp_act = torch.mul(sem_scores_tld, obj_rp) #(1, 64, 324, 324)

        #### crop 324 to in = 316 to get out = 132
        obj_rp_cr = obj_rp_act[:, :, 4:4 + DummyData_316.shape[2], 4:4 + DummyData_316.shape[3]] #Crop layer, (1, 64, 316, 316)

        n2_d0b = self.n2_d0b(obj_rp_cr) #(1, 64, 156, 156)
        n2_d0b_corrected = self.correction_layer(n2_d0b) #Corrected n2_d0b to fit into concat_n2_d2a_u1d layer
        concat_n2_d2a_u1d = torch.cat((n2_d0b_corrected, n1_d1c), dim=1)
        
        n2_d2a = self.n2_d2a(concat_n2_d2a_u1d)
        n1_d2c = self.n1_d2c()
        concat_n2_d2a_u2d = torch.cat((n2_d2a, n1_d2c), dim=1)

        return pred_cls

        '''
        pred_cls = pred_prob.argmax(1)
        return pred_cls
        '''

#MSRA weight initialization in caffe file
def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)


'''
model = DiskMaskNet().to(device)
print(model)
'''

'''
def train(model, device, train_data, train_target, optimizer ,epoch):
    model.train()
    for i in range(len(train_data)):
        print(train_data[i].shape)
        print(train_target[i].shape)
        optimizer.zero_grad()
        output = model(train_data[i])
        loss = nn.CrossEntropyLoss(output, train_target)
        loss.backward()
        optimizer.step()
        print(loss.item())
'''

def main():
    #Initialize model, optimizer and criterion
    model = DiskMaskNet().to(device)
    model.apply(weights_init)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss() #Same with caffe SoftmaxWithLoss layer (Softmax followed by multinomial logistic loss)
    
    #Declaring dummy data
    X = torch.rand((10, 3, 508, 508), device=device)
    label_2D_cls = torch.randint(2, (10, 324, 324), device=device)
    global gates 
    gates = torch.rand(1, 1, 324, 324).to(device=device)
    global DummyData_316 
    DummyData_316 = torch.zeros(1, 64, 316, 316).to(device=device)

    for epoch in range(30):
        for i in range(10):
            #y_pred = nn.Softmax(dim=1)(output)
            #y_pred = output.argmax(1)

            #Forward pass
            output = model(torch.unsqueeze(X[i], 0))
            loss = criterion(output, torch.unsqueeze(label_2D_cls[i], 0))

            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Loss: {}".format(loss.item()))

        #train(model = model, device = device, train_data=X, train_target=label_2D_cls, optimizer=optimizer, epoch = 0)

        #print(pred_prob.shape)
        #print(label_2D_cls.shape)

if __name__ == '__main__':
    main()

'''
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
'''


#####Version with batch size 1
'''
#X = torch.rand((10, 3, 508, 508), device=device)
#label_2D_cls = torch.randint(2, (10, 324, 324), device=device)


#y_pred = nn.Softmax(dim=1)(output)
    #y_pred = output.argmax(1)

    #Forward pass
    output = model(X)
    loss = criterion(output, label_2D_cls)

    #Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Loss: {}".format(loss.item()))

    #train(model = model, device = device, train_data=X, train_target=label_2D_cls, optimizer=optimizer, epoch = 0)

    #print(pred_prob.shape)
    #print(label_2D_cls.shape)
'''