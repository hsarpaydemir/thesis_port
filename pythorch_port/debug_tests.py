from torch import nn
import caffe

'''
temp_layer = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=2, padding=0), 
    nn.ReLU(),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
    nn.ReLU(),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=0),
    nn.ReLU(),
).to(device)
'''

def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
        
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + dilation * (kernel_size[0] - 1) + 1
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + dilation * (kernel_size[1] - 1) + 1
    
    return h, w

print("Convolution output: ", convtransp_output_shape(156, kernel_size=1, stride=2, pad=31))