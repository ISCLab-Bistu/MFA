# encoding: utf-8

import math

import torch
from torch import nn
from modeling.layer.non_local import Non_local
from modeling.layer.CPAM import CPAM
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)) ##############################channel缩减4倍
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

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

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


class ResNetSCP_1_NL(nn.Module):
    def __init__(self, last_stride=2, num_classes=751, block=Bottleneck, layers=[3, 4, 6, 3], non_layers=[0, 2, 3, 0]):

        super().__init__()

        self.inplanes = 64

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
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        self.row_pool = nn.AdaptiveAvgPool2d((4, 1))
        self.drop = nn.Dropout(0.25)

        # use conv to replace pool
        self.conv_pool1 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.conv_pool2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.conv_pool3 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.conv_pool4 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # classification part
        self.fc_cls1 = nn.Linear(2048, num_classes)
        self.fc_cls2 = nn.Linear(2048, num_classes)
        self.fc_cls3 = nn.Linear(2048, num_classes)
        self.fc_cls4 = nn.Linear(2048, num_classes)

        self.conv_bn1 = nn.BatchNorm2d(2048)
        self.conv_bn2 = nn.BatchNorm2d(2048)
        self.conv_bn3 = nn.BatchNorm2d(2048)
        self.conv_bn4 = nn.BatchNorm2d(2048)
        self.cbam = cbam(2048*4)
        self.se = SEModule(2048*4, 4)
        self.conv_last = Conv1x1(8192, 2048)
        self.conv_bn_last = nn.BatchNorm2d(2048)
        self.row1_bn = nn.BatchNorm1d(2048)
        self.row2_bn = nn.BatchNorm1d(2048)
        self.row3_bn = nn.BatchNorm1d(2048)
        self.row4_bn = nn.BatchNorm1d(2048)
        self.identity = nn.Identity()
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
        # x = self.conv_bn_last(x)
        row_x = self.row_pool(x)

        # row_x = self.conv_feature(row_x)
        N = x.size(0)
        row_f1 = row_x[:, :, 0].contiguous().view(N, -1)
        row_f2 = row_x[:, :, 1].contiguous().view(N, -1)
        row_f3 = row_x[:, :, 2].contiguous().view(N, -1)
        row_f4 = row_x[:, :, 3].contiguous().view(N, -1)
        row_f1 = self.row1_bn(row_f1).detach()
        row_f2 = self.row2_bn(row_f2).detach()
        row_f3 = self.row3_bn(row_f3).detach()
        row_f4 = self.row4_bn(row_f4).detach()
        
        
        x1 = self.conv_bn1(self.conv_pool1(x))
        x2 = self.conv_bn2(self.conv_pool2(x))
        x3 = self.conv_bn3(self.conv_pool3(x))
        x4 = self.conv_bn4(self.conv_pool4(x))

        x_all = torch.cat([x1, x2, x3, x4], dim=1)
        # return x1
        x1 = self.global_pool(x1).squeeze(3).squeeze(2)
        x2 = self.global_pool(x2).squeeze(3).squeeze(2)
        x3 = self.global_pool(x3).squeeze(3).squeeze(2)
        x4 = self.global_pool(x4).squeeze(3).squeeze(2)
        fmap['global'] = [x1, x2, x3, x4]
        fmap['global_cat'] = x_all
        fmap['local'] = [row_f1, row_f2, row_f3, row_f4]
        return fmap['global_cat']
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

