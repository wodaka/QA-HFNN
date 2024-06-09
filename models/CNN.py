from models.Mnist_model import MnistModel
import torch.nn as nn


class MyNetwork_dmnistn(nn.Module):
    """ 定义整个神经网络结构 """
    def __init__(self, class_num=10):
        super(MyNetwork_dmnistn, self).__init__()
        self.k = class_num
        self.class_layer = MnistModel()
        self.classi = nn.Linear(128, self.k)

    def forward(self, x):
        c_part = self.class_layer(x) #[batch,128]
        x = self.classi(c_part)
        return x

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #----------
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #
        # self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        #
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class ResidualBlock_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0, bias=False)
        self.conv4 = nn.Conv2d(128, 256 , kernel_size=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)


        # self.shortcut1 = nn.Conv2d(in_channels, 64, kernel_size=1, padding=0)
        # self.bnst1 = nn.BatchNorm2d(64)
        #
        # self.shortcut2 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        # self.bnst2 = nn.BatchNorm2d(128)

        self.skipconnec = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, bias=False)
        self.bnsk = nn.BatchNorm2d(256)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        # out += self.bnst1(self.shortcut1(residual))

        out = self.relu(self.bn3(self.conv3(out)))

        # out += self.bnst2(self.shortcut2(residual1))

        out = self.relu(self.bn4(self.conv4(out)))

        out += self.bnsk(self.skipconnec(residual))

        out = self.relu(out)
        return out


class MyNetwork_JAFFE(nn.Module):
    def __init__(self):
        super(MyNetwork_JAFFE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # self.res1 = ResidualBlock_4(64, 256)
        self.res1 = ResidualBlock(64, 64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # self.res2 = ResidualBlock_4(128, 256)
        self.res2 = ResidualBlock(128, 128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)  # Assuming 7 classes for facial expressions
        self.fc3 = nn.Linear(512, 7)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))

        x = torch.relu(self.res1(x))

        # print(x.shape)
        #
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))

        x = torch.relu(self.res2(x))

        x = self.pool(torch.relu(self.conv5(x)))
        x = torch.relu(self.conv6(x))

        # print(x.shape)

        x = x.view(-1, 512 * 4 * 4)
        self.dropout(x)
        x = torch.relu(self.fc1(x))
        self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyNetwork_JAFFE_DNN(nn.Module):
    def __init__(self):
        super(MyNetwork_JAFFE_DNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.res1 = ResidualBlock(64, 64)
        # self.res1 = ResidualBlock_4(64, 256)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.res2 = ResidualBlock(128, 128)
        # self.res2 = ResidualBlock_4(128, 256)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)  # Assuming 7 classes for facial expressions
        # self.fc3 = nn.Linear(512, 256)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # print(x.shape)
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))

        x = torch.relu(self.res1(x))

        # print(x.shape)
        #
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.conv4(x))

        x = torch.relu(self.res2(x))

        x = self.pool(torch.relu(self.conv5(x)))
        x = torch.relu(self.conv6(x))

        # print(x.shape)

        x = x.view(-1, 512 * 4 * 4)
        self.dropout(x)
        x = torch.relu(self.fc1(x))
        # self.dropout(x)
        x = self.fc2(x)

        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


