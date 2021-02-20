import torch
import torch.nn as nn
import numpy as np

import os
import pickle
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split




def conv(in_planes, out_planes, kernel_size=8, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size,
        stride=stride,
        padding=int((kernel_size - 1) / 2),
        bias=False)

class MyDataset(Dataset):
    def __init__(self, datas, labels):

        self.data = datas
        self.label = labels
        print(len(self.label))

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)



class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_planes, planes, kernel_size, 1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = 1
        #
        self.conv2 = conv(planes, planes, kernel_size, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        #
        self.conv3 = conv(planes, planes, kernel_size, 1)
        self.bn3 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResEncoder(nn.Module):
    def __init__(self, block, kernel_size, num_classes=6, in_planes=10):  # block means BasicBlock
        self.in_planes = in_planes
        super(ResEncoder, self).__init__()
        self.layer1 = self._make_layer(block, kernel_size[0], 64)  # 128)
        self.layer2 = self._make_layer(block, kernel_size[1], 128)  # 256)
        self.layer3 = self._make_layer(block, kernel_size[2], 128)  # 512)

        self.pool = nn.MaxPool1d(2, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)
        # self.inm = nn.InstanceNorm1d(256)

    def _make_layer(self, block, kernel_size, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.in_planes, planes, kernel_size, stride, downsample))
        self.in_planes = planes
        # for i in range(1,blocks):
        #       layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)  # batch,512,60
        x = x[:, int(x.size(1) / 2):, :].mul(self.softmax(x[:, :int(x.size(1) / 2), :]))  # batch,256,60
        x = x.sum(2)
        x = x.view(x.size(0), -1)
        # x = self.inm(x)
        x = self.fc(x)
        return x

# net = ResEncoder(BasicBlock,[9,5,3],7,17)  模型定义， [9,5,3]分别对应1 2 3层的kernalsize  7是class， 17是channel
# (number, channel, length)  (12807, 17, 60)这是数据的输入格式
