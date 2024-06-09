import torch
import torch.nn as nn
import sys
from network_layers.FuzzyLayers import Q_Model
import torchquantum as tq
from models.CNN_model import classical_layer
from network_layers.NetworkLayer import classical_part_layer

class Q_Model(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 3

            # gates with trainable parameters
            self.rz00 = tq.RZ(has_params=True, trainable=True)
            self.ry00 = tq.RY(has_params=True, trainable=True)
            self.rz01 = tq.RZ(has_params=True, trainable=True)

            self.rz10 = tq.RZ(has_params=True, trainable=True)
            self.ry10 = tq.RY(has_params=True, trainable=True)
            self.rz11 = tq.RZ(has_params=True, trainable=True)

            self.rz20 = tq.RZ(has_params=True, trainable=True)
            self.ry20 = tq.RY(has_params=True, trainable=True)
            self.rz21 = tq.RZ(has_params=True, trainable=True)

        def forward(self, device: tq.QuantumDevice):

            self.rz00(device, wires=0)
            self.ry00(device, wires=0)
            self.rz01(device, wires=0)

            self.rz10(device, wires=1)
            self.ry10(device, wires=1)
            self.rz11(device, wires=1)

            self.rz20(device, wires=2)
            self.ry20(device, wires=2)
            self.rz21(device, wires=2)


    def __init__(self):
        super().__init__()
        self.n_wires = 3
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
        ])

        self.q_layer = self.QLayer()
        self.q_layer1 = self.QLayer()
        self.q_layer2 = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)


    def forward(self, x, use_qiskit=False):

        bsz = x.shape[0]
        # x_shape_1 = x.shape[1]
        # x_shape_2 = x.shape[2]
        # x_shape_3 = x.shape[3]

        x = x.permute(0, 2, 3, 1)
        # print(x.shape)

        x = x.reshape(x.shape[0]*32*32,3)
        # print(x.shape)

        device = tq.QuantumDevice(n_wires=3, bsz=x.shape[0], device='cuda')

        #re-upload ansaz

        self.encoder(device, x)
        self.q_layer(device)

        self.encoder(device, x)
        self.q_layer1(device)

        self.encoder(device, x)
        self.q_layer2(device)

        x = self.measure(device)
        # print(x.shape)
        # x = x.view(bsz, 10)
        x = x.view(bsz,32,32,3)
        qout = x.permute(0, 3, 1, 2) #[batch,3,32,32]

        return qout
class QFuzzyLayer(nn.Module):
    """ 高斯层，包含 k 个高斯函数 """
    def __init__(self, k): #k是类别
        super(QFuzzyLayer, self).__init__()
        self.qfuzziers = nn.ModuleList([Q_Model() for _ in range(k)])

    def forward(self, x):
        outputs = []
        for qfuzzier in self.qfuzziers:
            outputs.append(qfuzzier(x))
        return torch.stack(outputs, dim=-1)

class MyNetwork(nn.Module):
    """ 定义整个神经网络结构 """
    def __init__(self, class_num=10):
        super(MyNetwork, self).__init__()
        self.k = class_num

        self.class_layer = classical_layer()

        self.qfuzzy_layer = QFuzzyLayer(class_num)
        self.fuzzy_rule = nn.Sequential(nn.Dropout(0.4),  # 两层fc效果还差一些
                                        nn.Linear(3*32*32, 1), )  # fc，最终Cifar10输出是10类

        self.classifier = nn.Sequential(nn.Linear(20, 10), )  # fc，最终Cifar10输出是10类

    def forward(self, x):
        #[batch_size,3,32,32]


        batch_size = x.size(0)
        c_part = self.class_layer(x)

        # x = x.view(-1)  # 改变 x 的形状以匹配高斯层
        x = self.qfuzzy_layer(x)
        x = x.view(batch_size, -1, self.k)  # 调整输出形状为 [batch_size, 3*32*32, k]


        x = x.permute(0,2,1)
        # print(x.shape)
        # fuzzy_rule_output = torch.prod(x, dim=1)
        fuzzy_rule_output = self.fuzzy_rule(x)

        # print(fuzzy_rule_output[0])

        # print(fuzzy_rule_output.shape)
        x = fuzzy_rule_output.view(batch_size,10)
        # print(x.shape)
        fusion_output = torch.cat((c_part, x), dim=1)

        x = self.classifier(fusion_output)

        return x


