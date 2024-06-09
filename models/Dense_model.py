import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

hidden_dim = 256
class DenseNet(torch.nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.l1 = torch.nn.Linear(200, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l4 = torch.nn.Linear(hidden_dim, 15)
        # self.l5 = torch.nn.Linear(256, 15)

        self.dropout = torch.nn.Dropout(p=0.4)
        self.acti = torch.nn.ReLU()
        self.acti2 = torch.nn.Sigmoid()

    def forward(self, x):
        # x = x.view(-1, 784)
        x = self.dropout(x)
        x = self.acti(self.l1(x))
        x = self.dropout(x)
        x = self.acti(self.l2(x))
        x = self.dropout(x)
        x = self.acti(self.l3(x))
        return x
        # return self.l3(x)


class DenseNet_S(torch.nn.Module):
    def __init__(self):
        super(DenseNet_S, self).__init__()
        self.l1 = torch.nn.Linear(200, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l3 = torch.nn.Linear(hidden_dim, 256)
        self.l4 = torch.nn.Linear(hidden_dim, 15)
        # self.l5 = torch.nn.Linear(256, 15)

        self.dropout = torch.nn.Dropout(p=0.4)
        self.acti = torch.nn.ReLU()
        self.acti2 = torch.nn.Sigmoid()

    def forward(self, x):
        # x = x.view(-1, 784)
        x = self.dropout(x)
        x = self.acti(self.l1(x))
        x = self.dropout(x)
        x = self.acti(self.l2(x))
        x = self.dropout(x)
        x = self.l3(x)
        return x

hidden_dim = 256

class DenseNet_15scene(torch.nn.Module):
    def __init__(self, class_num=15):
        super(DenseNet_15scene, self).__init__()
        self.class_layer = DenseNet()

        self.classifier = nn.Sequential(nn.Dropout(0),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, class_num),
                                        )  # fc，最终Cifar10输出是10类

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.uniform_(m.weight, a=-1 / torch.sqrt(torch.tensor(class_num * 2)),
                              b=1 / torch.sqrt(torch.tensor(class_num * 2)))
                if m.bias is not None:
                    init.zeros_(m.bias)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        c_part = self.class_layer(x)

        x = self.classifier(c_part)

        return x