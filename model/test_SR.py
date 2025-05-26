import math
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import random
import numpy as np


from PIL import Image as m

import torch
import torch.nn as nn
import torch.nn.functional as F




# Super resolution model
################################################
class SRBranch(nn.Module):
    def __init__(self, n_features, up_scale, bn=True, act='relu'):
        super(SRBranch, self).__init__()
        m = []
        if (up_scale & (up_scale - 1)) == 0:  # Is scale = 2^n?
            for i in range(int(math.log(up_scale, 2))):
                m.append(nn.ConvTranspose2d(n_features, n_features // 2,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1,
                                            bias=True))
                m.append(nn.Conv2d(n_features // 2, n_features // 2, kernel_size=1))
                if bn:
                    m.append(nn.BatchNorm2d(n_features // 2))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                n_features = n_features // 2
        else:
            raise NotImplementedError
        self.upsample = nn.Sequential(*m)

        self.classfier = nn.Conv2d(n_features, 3, kernel_size=1)
    def forward(self, x):
        feature = self.upsample(x)
        sr = self.classfier(feature)

        return feature, sr



model = SRBranch(512,32)
input = torch.randn(2,512,14,14)
feature,sr = model(input)
print(feature.shape)
print(sr.shape)