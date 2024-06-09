import torch.nn as nn
import torch

import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(320, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

        self.linearn = nn.Linear(320, 128)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        # nn.Tanh()
        # self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x))) #[batch,10,12,12]
        x = self.relu(self.maxpool2(self.conv2(x))) #[batch,20,4,4]
        x = x.view(x.size(0), -1)
        # x = self.relu(self.linear1(x))
        # x = self.relu(self.linear2(x))
        # x = self.linear3(x)
        # x = self.dropout(x) #如果训练集精度和测试集精度相差较多，则加上drop-out层
        x = self.linearn(x)
        return x


# 设计模型
class DenseNet(torch.nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
