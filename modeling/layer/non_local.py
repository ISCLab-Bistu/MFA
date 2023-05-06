# encoding: utf-8

import torch
from torch import nn


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class Mul_Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Mul_Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        self.g2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=3, stride=1, padding=1)
        self.g3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=5, stride=1, padding=2)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.theta2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=3, stride=1, padding=1)
        self.theta3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=5, stride=1, padding=2)
        self.phi1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.phi2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=3, stride=1, padding=1)
        self.phi3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        '''
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        '''
        batch_size = x.size(0)
        g_x = self.g1(x)+self.g2(x)+self.g3(x)
        g_x = g_x.view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta1(x)+self.theta2(x)+self.theta3(x)
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi1(x)+self.phi2(x)+self.phi3(x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z
import torch.nn.functional as F
class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 池化层，将每一个通道的宽和高都变为 1 (平均值)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 先降低维
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            # 再升维
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # y是权重
        return x * y.expand_as(x)


class SE_Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(SE_Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.se = SEAttention(channel=self.in_channels)
    def forward(self, x):
        '''
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = self.se(W_y + x)
        return z