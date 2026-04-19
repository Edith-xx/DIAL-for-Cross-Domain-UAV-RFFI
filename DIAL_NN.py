import torch
import math
import torch.nn as nn
from inception import InceptionModule
from functions import ReverseLayerF
import random
import numpy as np
def manifold(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]#tensor的第一个维度，这里是batch_size的大小
    if use_cuda:
        index = torch.randperm(batch_size).cuda()#生成随机排列的索引，用于从批次中选择要混合的样本，randperm用于生成从0~batch_size之间随机排列的整数，岁哦及选择样本用于组合
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]#原始输入与索引选定的x的组合
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
class WaveletBasedResidualAttentionNet(nn.Module):
    def __init__(self, input_channels=4, depth=1, ratio=4, width=64, alpha=0.01):
        super().__init__()
        self.depth = depth
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7,  padding=3),
            nn.LeakyReLU(negative_slope=alpha)
        )
        self.inception_module = InceptionModule(in_channels=width, width=width, ratio=ratio, alpha=alpha)
        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels=width * depth, out_channels=width, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Conv2d(in_channels=width, out_channels=4, kernel_size=3, padding=1),
        )
    def forward(self, x, target):
        out = self.feature_extraction(x)
        outs = []
        for _ in range(self.depth):
            residual = out
            out = self.inception_module(out)
            out += residual
            outs.append(out)
        out = self.final_layers(torch.cat(outs, dim=1))
        if self.training:
            feature, y_a, y_b, lam = manifold(out, target, 0.5)
        else:
            feature, y_a, y_b, lam = out, target, target, 1
        return feature, y_a, y_b, lam
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = WaveletBasedResidualAttentionNet(1, 2, 4, 64, 0.01)
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.LazyLinear(1000))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(1000))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(1000, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 6))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))#将输入n维向量缩放到（0，1）之间且和为1

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.LazyLinear(100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
    def forward(self, input_data, target, alpha):
        feature, y_a, y_b, lam = self.feature(input_data, target)
        feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output, y_a, y_b, lam
