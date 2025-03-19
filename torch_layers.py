import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
# local packages - use conda develop
import torch_utils

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device is: " + str(device))

class PiecewiseReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.ReLU()
        
    def forward(self, x):
        return x + self.f((x - 1) / 2) + self.f((-1 - x) / 2)

class BatchNormConv(nn.Module):
    "Performs Batch Normalization followed by a convolution"
    def __init__(self, in_channels, out_channels, kernel_size, hyp_conv=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.hyp_conv = hyp_conv
        if self.hyp_conv:
            self.conv = nel.HyperConv2DFromDense(out_channels, kernel_size, padding='same')
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')

    def forward(self, x):
        if self.hyp_conv:
            x, hyp_tensor = x
            x = self.bn(x)
            x = self.conv((x, hyp_tensor))
        else:
            x = self.bn(x)
            x = self.conv(x)
        return x

class Mix(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix_param = nn.Parameter(torch.rand(1))
        self.s = nn.Sigmoid()

    def forward(self, x):
        A, B = x
        sig_mix = self.s(self.mix_param)
        return sig_mix * A + (1 - sig_mix) * B

class Interleaved(nn.Module):
    def __init__(self, N, kernel_size=5, num_convs=1, shift=False, hyp_conv=False, device=device):
        super().__init__()
        self.features = N
        self.kernel_size = kernel_size
        self.num_convs = num_convs
        self.shift = shift
        self.hyp_conv = hyp_conv
        self.img_mix = Mix().to(device)
        self.freq_mix = Mix().to(device)
        self.img_bnconvs = [BatchNormConv(N, N, kernel_size=kernel_size, hyp_conv=hyp_conv).to(device) for i in range(num_convs)]
        self.freq_bnconvs = [BatchNormConv(N, N, kernel_size=kernel_size, hyp_conv=hyp_conv).to(device) for i in range(num_convs)]
        self.relu = nn.ReLU()
        self.p_relu = PiecewiseReLU()
        
    def forward(self, x):
        if self.hyp_conv:
            img_in, freq_in, hyp_tensor = x
        else:
            img_in, freq_in = x
        img_in_as_freq = torch_utils.convert_channels_to_freq(img_in)
        freq_in_as_img = torch_utils.convert_channels_to_image(freq_in)
        #print("shapes of img_in and freq_in are: " + str(img_in.shape) + " and " + str(freq_in.shape))
        #print("shapes of img_in_as_freq and freq_in_as_img are: " + str(img_in_as_freq.shape) + " and " + str(freq_in_as_img.shape))
        img_feat = self.img_mix([img_in, freq_in_as_img])
        k_feat = self.freq_mix([freq_in, img_in_as_freq])
        #print("shapes of img_feat and k_feat are: " + str(img_feat.shape) + " and " + str(k_feat.shape))
        for i in range(self.num_convs):
            # process image-space features
            if self.shift:
                img_feat = torch.fft.ifftshift(img_feat, dim=(2,3))
            if self.hyp_conv:
                img_conv = self.img_bnconvs[i]((img_feat, hyp_tensor))
            else:
                img_conv = self.img_bnconvs[i](img_feat)
            img_feat = self.relu(img_conv)
            # process frequency-space features
            if self.shift:
                k_feat = torch.fft.ifftshift(k_feat, dim=(2,3))
            if self.hyp_conv:
                k_conv = self.freq_bnconvs[i]((k_feat, hyp_tensor))
            else:
                k_conv = self.freq_bnconvs[i](k_feat)
            k_feat = self.p_relu(k_conv)
        return (img_feat, k_feat)