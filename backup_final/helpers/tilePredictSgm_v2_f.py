import math
import numpy as np
import matplotlib.pyplot as plt
from diskmaskport_single_gpu import DiskMaskNet
from helpers.cropBlob_f import cropBlob_f
from torch import nn
import torch
import skfuzzy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
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

        enco_intermediate = {'n1_d1c': n1_d1c, 'n1_d2c': n1_d2c, 'n1_d3c': n1_d3c, 'n1_d4c': n1_d4c, 'obj_rp' : obj_rp}
        return pred_cls, pred_semseg, enco_intermediate
#MSRA weight initialization in caffe file
def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.zeros_(m.bias.data)

def tilePredictSgm_v2_f(data, opts):
    model = DiskMaskNet().to(device)
    #model.apply(weights_init)
    model.load_state_dict(torch.load('/home/haydemir/Desktop/connectivity/thesis/pythorch_port/fixed_tensor_model_zerograd_15000.pt'))
    model.eval()

    debugMode = 0
    input_size = opts['padInput']
    output_size = opts['padOutput']
    ntiles_x = math.ceil(data.shape[0] / output_size[0])
    ntiles_y = math.ceil(data.shape[1] / output_size[1])

    print('Insize: ', input_size)
    print('Outsize: ', output_size)

    # create padded volume with maximal border

    border = (np.around(input_size - output_size) / 2).astype(int)
    
    if len(data.shape) == 3:
        data = data[:, :, :, np.newaxis]
    paddedFullVolume = np.zeros((data.shape[0] + 2 * border[0], 
                                data.shape[1] + 2 * border[1],
                                data.shape[2],
                                data.shape[3]), dtype=np.single)

    paddedFullVolume[border[0]: border[0] + data.shape[0], border[1]:border[1] + data.shape[1], :, :] = data

    if opts['padding'] == 'mirror':
        xpad = border[0]
        xfrom = border[0] + 1
        xto = border[0] + data.shape[0]
        paddedFullVolume[0:xfrom - 1, :, :] = paddedFullVolume[xfrom + xpad - 1:xfrom - 1:-1, :, :]
        paddedFullVolume[xto:, :, :] = paddedFullVolume[xto - 2:xto - xpad - 2:-1, :, :]
        
        ypad = border[1]
        yfrom = border[1] + 1
        yto = border[1] + data.shape[1]
        paddedFullVolume[:, 0:yfrom - 1, :] = paddedFullVolume[:, yfrom + ypad - 1:yfrom - 1:-1, :]
        paddedFullVolume[:, yto:, :] = paddedFullVolume[:, yto - 2: yto - ypad - 2:-1, :]

    xcoord = output_size[0]*ntiles_x-data.shape[0]
    ycoord = output_size[1]*ntiles_y-data.shape[1]
    paddedFullVolume = np.pad(np.squeeze(paddedFullVolume), ((xcoord, xcoord), (ycoord, ycoord), (0, 0)), mode='symmetric')
    paddedFullVolume = paddedFullVolume[xcoord:, ycoord:]
    scores = [None] * len(opts['expectedPredData'])
    for num in range(data.shape[3]):
        scores_t = [None] * len(opts['expectedPredData'])
        print('Processing image {}'.format(num))
        for yi in range(ntiles_y):
            if ((yi * output_size[1] + 1) > paddedFullVolume.shape[1]):
                print('----------> skip yi = {}  (out of bounds)'.format(yi))
                continue
            for xi in range(ntiles_x):
                if ((xi * output_size[0] + 1) > paddedFullVolume.shape[0]):
                    print('----------> skip xi = {}, yi = {}  (out of bounds)'.format(xi, yi))
                    continue
                else:
                    print('-----> tile [{}, {}]'.format(yi + 1, xi + 1))
                paddedInputSlice = np.zeros((input_size[0], input_size[1], data.shape[2]), dtype=np.single)
                validReg = []
                validReg.append(min(input_size[0], paddedFullVolume.shape[0] - xi * output_size[0]))
                validReg.append(min(input_size[1], paddedFullVolume.shape[1] - yi * output_size[1]))
                paddedInputSlice[:validReg[0], :validReg[1], :] = paddedFullVolume[xi * output_size[0]:xi * output_size[0] + validReg[0],
                                                                            yi * output_size[1]:yi * output_size[1] + validReg[1], :]
                #solver.test_nets(1).forward({paddedInputSlice});
                mock_gates = torch.ones((324, 324)).to(device)
                pred_cls, pred_semseg, expected_pred_data = model(torch.permute(torch.tensor(paddedInputSlice[:, :, :3]).unsqueeze(0), (0, 3, 1, 2)).to(device), mock_gates)

                '''
                for c_i in range(len(opts['expectedPredData'])):
                    pred_cls, pred_semseg, expected_pred_data = model(opts['expectedPredData'][c_i], mock_gates)
                '''
                
                pred_cls = paddedInputSlice[:, :, 3]
                pred_cls = cropBlob_f(pred_cls, [opts['netSize_1_l'][0], opts['netSize_1_l'][1]])
                un = np.unique(pred_cls)
                un = np.array(un[un != 0], dtype=int)

                if yi == 0 and xi == 0:
                    scoreSlice = pred_semseg
                    #scores_tmp = np.zeros((data.shape[0], data.shape[1], scoreSlice.shape[1], scoreSlice.shape[0]), dtype=np.single)
                    scores_tmp = np.zeros((data.shape[0], data.shape[1], np.unique(data[:, :, 3]).astype(int)[-1], scoreSlice.shape[0]), dtype=np.single)
                else:
                    scores_tmp = scores_t[0]

                objN = 0
                if len(un) != 0:
                    for un_i in un:
                        m_bin = pred_cls == un_i

                        pred_cls2, pred_semseg2, expected_pred_data2 = model(torch.permute(torch.tensor(paddedInputSlice[:, :, :3]).unsqueeze(0), (0, 3, 1, 2)).to(device), torch.from_numpy(np.transpose(m_bin, (3, 2, 0, 1))).to(device))

                        scoreSlice = skfuzzy.sigmf(np.transpose(pred_semseg2.cpu().detach().numpy(), (2, 3, 1, 0)), 0, 1)
                        validReg[0] = min(output_size[0], scores_tmp.shape[0] - xi * output_size[0])
                        validReg[1] = min(output_size[1], scores_tmp.shape[1] - yi * output_size[1])
                        xfromto = [xi * output_size[0], xi * output_size[0] + validReg[0]]
                        yfromto = [yi * output_size[1], yi * output_size[1] + validReg[1]]
                        objN += 1
                        scores_tmp[xfromto[0]:xfromto[1], yfromto[0]:yfromto[1], un_i - 1] = scoreSlice[:validReg[0], :validReg[1], 0]
                        print(pred_cls)

                scores_t[0] = scores_tmp
                if debugMode > 0:
                    plt.title('Image: {} / {} Tile: {} / {} / {} [{}, {}]'.format(num + 1, data.shape[3], yi*ntiles_x + xi + 1, ntiles_x, ntiles_y, yi+1, xi+1))
                    plt.imshow(scores_tmp[:, :, 1, num] - scores_tmp[:, :, 0, num])
                    plt.show()

        '''
        for cl_i in range(len(scores_t)):
            tmp = scores_t[cl_i]
            scores_cl = scores[cl_i]
            if scores_cl == None:
                scores_cl = tmp.copy()
            else:
                scores_cl = np.concatenate((scores_cl, tmp), axis=3)
            scores[cl_i] = scores_cl
        '''
        tmp = scores_t[0]
        scores_cl = scores[0]
        if scores_cl == None:
            scores_cl = tmp.copy()
        else:
            scores_cl = np.concatenate((scores_cl, tmp), axis=3)
        scores[0] = scores_cl

    return scores