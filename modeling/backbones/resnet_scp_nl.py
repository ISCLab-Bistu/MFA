# encoding: utf-8

import math

import torch
from torch import nn
from modeling.layer.non_local import Non_local
from modeling.layer.CPAM import CPAM
from modeling.layer.da_att import DUAM
from ..layer import GeneralizedMeanPoolingP
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam(nn.Module):
    def __init__(self, in_planes):
        super(cbam, self).__init__()
        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetSCPNL(nn.Module):
    def __init__(self, last_stride=2, num_classes=751, block=Bottleneck, layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0]):

        super().__init__()
        self.inplanes = 64
        self.outdim = 2048
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

        self.NL_1 = nn.ModuleList(
            [CPAM(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [CPAM(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [CPAM(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [CPAM(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        self.row_pool = nn.AdaptiveAvgPool2d((4, 1))
        self.drop = nn.Dropout(0.1)
        # self.CPAM = CPAM(2048)
        # self.Nonl = Non_local(2048)
        # use conv to replace pool

        # self.conv_last = nn.Conv2d(2048, self.outdim, kernel_size=1, stride=1, padding=0)
        self.conv_pool1 = nn.Conv2d(self.outdim, self.outdim, kernel_size=1, stride=1, padding=0)
        self.conv_pool2 = nn.Conv2d(self.outdim, self.outdim, kernel_size=1, stride=1, padding=0)
        self.conv_pool3 = nn.Conv2d(self.outdim, self.outdim, kernel_size=1, stride=1, padding=0)
        self.conv_pool4 = nn.Conv2d(self.outdim, self.outdim, kernel_size=1, stride=1, padding=0)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gmpool = GeneralizedMeanPoolingP()
        self.Sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # classification part
        self.fc_cls1 = nn.Linear(self.outdim, num_classes)
        self.fc_cls2 = nn.Linear(self.outdim, num_classes)
        self.fc_cls3 = nn.Linear(self.outdim, num_classes)
        self.fc_cls4 = nn.Linear(self.outdim, num_classes)
        self.fc_cls = nn.Linear(4*self.outdim, num_classes)
        self.conv_bn1 = nn.BatchNorm2d(self.outdim)
        self.conv_bn2 = nn.BatchNorm2d(self.outdim)
        self.conv_bn3 = nn.BatchNorm2d(self.outdim)
        self.conv_bn4 = nn.BatchNorm2d(self.outdim)
        self.cbam = cbam(self.outdim)
        # self.conv_bn = nn.BatchNorm2d(self.outdim*4)
        self.identity = nn.Identity()
        self.bottleneck = nn.BatchNorm1d(self.outdim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.relu = nn.ReLU()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)
        fmap = dict()
        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
        # return x                                            #use grad_cam
        # return self.gap(x).view(x.size(0),-1)             #use grad_cam
        # new feature
        if self.outdim != 2048:
            x = self.conv_last(x)
        row_x = self.row_pool(x)
        # row_x = self.conv_feature(row_x)
        N = x.size(0)
        row_f1 = row_x[:, :, 0].contiguous().view(N, -1)
        row_f2 = row_x[:, :, 1].contiguous().view(N, -1)
        row_f3 = row_x[:, :, 2].contiguous().view(N, -1)
        row_f4 = row_x[:, :, 3].contiguous().view(N, -1)

        x1 = self.conv_pool1(x)#去掉cbam,conv_bn1,和局部分支统一；不服从高斯分布，计划考虑crossentrophy代替mse
        x2 = self.conv_pool2(x)
        x3 = self.conv_pool3(x)
        x4 = self.conv_pool4(x)

        # new feature
        conv_f1 = self.global_pool(self.conv_bn1(x1)).squeeze(3).squeeze(2)
        conv_f2 = self.global_pool(self.conv_bn2(x2)).squeeze(3).squeeze(2)
        conv_f3 = self.global_pool(self.conv_bn3(x3)).squeeze(3).squeeze(2)
        conv_f4 = self.global_pool(self.conv_bn4(x4)).squeeze(3).squeeze(2)
        fmap['global'] = [conv_f1, conv_f2, conv_f3, conv_f4]
        # return x3
        # x = x1 + x2 + x3 + x4
        # sumx = torch.sum(x,dim=1)
        # p1 = sns.heatmap(sumx.squeeze(0).detach().numpy(), annot=True)
        # s1 = p1.get_figure()
        # s1.savefig('./x.jpg')
        # plt.show()
        # return x
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        # x_cat = x4
        return torch.sum(x_cat, dim=1).unsqueeze(0)

        # classification
        # conv_cat = self.global_pool(self.conv_bn(x_cat)).squeeze(3).squeeze(2)
        # bnn_conv_cat = self.bottleneck(conv_cat)
        # s_cat = self.fc_cls(conv_cat)
        s1 = self.fc_cls1(conv_f1) #可用考虑去掉drop
        s2 = self.fc_cls2(conv_f2)
        s3 = self.fc_cls3(conv_f3)
        s4 = self.fc_cls4(conv_f4)
        fmap['local'] = [row_f1, row_f2, row_f3, row_f4]

        fmap['cls'] = [s1, s2, s3, s4]
        # fmap['cls'] = [s_cat]
        # fmap['global_cat'] = conv_cat
        return fmap

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

