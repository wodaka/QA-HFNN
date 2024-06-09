import torch
import torch.nn as nn
import sys
from network_layers.FuzzyLayers import Q_Model
import torchquantum as tq
from models.CNN_model import classical_layer
from network_layers.NetworkLayer import classical_part_layer

class MyModel(nn.Module):
    def __init__(self, fuzzy_feature_num, classical_output_feature_num=4 * 512):
        super(MyModel, self).__init__()

        #原文中设置为类别数量
        num_class = fuzzy_feature_num

        self.fuzzy_layer = Q_Model(256) #batch 256
        # self.classical_layer = classical_part_layer(input_feature_num,classical_output_feature_num)
        # self.classical_layer = classical_layer()

        # self.dense1_af_fusion = nn.Linear(fuzzy_feature_num+classical_output_feature_num, 128)
        # self.output_layer = nn.Linear(128, num_class)

        self.dense1 = nn.Linear(3*32*32, 8)  # fc，最终Cifar10输出是10类
        # self.classifier = nn.Linear(fuzzy_feature_num + 10, 10) # fc，最终Cifar10输出是10类
        self.classifier = nn.Linear(8, 10)  # fc，最终Cifar10输出是10类
        # self.classifier = nn.Linear(10, 10)  # fc，最终Cifar10输出是10类

    def forward(self, x):
        x_flat = x.view(x.shape[0],-1)
        x_flat = self.dense1(x_flat)
        # x is a list of tensors
        fuzz_member = self.fuzzy_layer(x_flat)
        #fuzzy规则：AND规则
        # fuzzy_rule_output = torch.prod(fuzz_member, dim=1)

        #经典网络部分
        # classical_output = self.classical_layer(x)

        #特征聚合
        # fusion_output  = torch.cat((fuzzy_rule_output,classical_output), dim=1)
        # fusion_output  =  fuzzy_rule_output

        #聚合后的dense
        # daff_layer_1_out = torch.sigmoid(self.dense1_af_fusion(fusion_output))
        # #分类器 -> num_class
        # output = torch.softmax(self.output_layer(daff_layer_1_out), dim=1)
        # output = self.classifier(fusion_output)
        # output = torch.sigmoid(self.classifier(fusion_output))
        output = torch.sigmoid(self.classifier(fuzz_member))

        return output

class Q_Model(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1

            # gates with trainable parameters
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)

        def forward(self, device: tq.QuantumDevice):

            self.rz0(device, wires=0)
            self.ry0(device, wires=0)
            self.rz1(device, wires=0)


    def __init__(self):
        super().__init__()
        self.n_wires = 10
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
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

        x = x.view(-1,1)
        # print(x)
        #做一个actan，使得输入数据在一个周期范围内。
        # x = torch.arctan(x) #效果不好 Ry门的周期是pi，需要把输入变换至输入的范围
        # x = torch.pi * x

        device = tq.QuantumDevice(n_wires=1, bsz=x.shape[0], device='cuda')

        #re-upload ansaz
        self.encoder(device, x)
        self.q_layer(device)

        self.encoder(device, x)
        self.q_layer1(device)
        #
        # self.encoder(device, x)
        # self.q_layer2(device)

        # self.encoder(device, x)
        # self.q_layer3(device)
        #
        # self.encoder(device, x)
        # self.q_layer4(device)

        x = self.measure(device)
        # x = x.view(bsz, 10)
        qout = (x+1)/2

        # print(qout)

        return qout

class Q_Model1(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1

            # gates with trainable parameters
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)

        def forward(self, device: tq.QuantumDevice):

            self.rz0(device, wires=0)
            self.ry0(device, wires=0)
            self.rz1(device, wires=0)


    def __init__(self):
        super().__init__()
        self.n_wires = 10
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
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

        x = x.view(-1,1)
        # print(x)
        #做一个actan，使得输入数据在一个周期范围内。
        # x = torch.arctan(x) #效果不好 Ry门的周期是pi，需要把输入变换至输入的范围
        # x = torch.pi * x

        device = tq.QuantumDevice(n_wires=1, bsz=x.shape[0], device='cuda')

        #re-upload ansaz
        self.encoder(device, x)
        self.q_layer(device)

        # self.encoder(device, x)
        # self.q_layer1(device)
        # #
        # self.encoder(device, x)
        # self.q_layer2(device)

        # self.encoder(device, x)
        # self.q_layer3(device)
        #
        # self.encoder(device, x)
        # self.q_layer4(device)

        x = self.measure(device)
        # x = x.view(bsz, 10)
        qout = (x+1)/2

        # print(qout)

        return qout


class Q_Model_muti_qubit(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 1

            # gates with trainable parameters
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)

        def forward(self, device: tq.QuantumDevice):

            self.rz0(device, wires=0)
            self.ry0(device, wires=0)
            self.rz1(device, wires=0)


    def __init__(self):
        super().__init__()
        self.n_wires = 10
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
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

        x = x.view(-1,1)

        device = tq.QuantumDevice(n_wires=1, bsz=x.shape[0], device='cuda')

        #re-upload ansaz
        self.encoder(device, x)
        self.q_layer(device)

        self.encoder(device, x)
        self.q_layer1(device)
        #
        self.encoder(device, x)
        self.q_layer2(device)

        # self.encoder(device, x)
        # self.q_layer3(device)
        #
        # self.encoder(device, x)
        # self.q_layer4(device)

        x = self.measure(device)
        # x = x.view(bsz, 10)
        qout = (x+1)/2

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

class QFuzzyLayer1(nn.Module):
    """ 高斯层，包含 k 个高斯函数 """
    def __init__(self, k): #k是类别
        super(QFuzzyLayer1, self).__init__()
        self.qfuzziers = nn.ModuleList([Q_Model1() for _ in range(k)])

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

        x = x.view(-1)  # 改变 x 的形状以匹配高斯层
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

        # x = c_part

        return x

from models.Mnist_model import MnistModel
class MyNetwork_dmnist(nn.Module):
    """ 定义整个神经网络结构 """
    def __init__(self, class_num=10):
        super(MyNetwork_dmnist, self).__init__()
        self.k = class_num

        self.class_layer = MnistModel()

        self.qfuzzy_layer = QFuzzyLayer(class_num)
        # self.fuzzy_rule = nn.Sequential(nn.Dropout(0.4),  # 两层fc效果还差一些
        #                                 nn.Linear(1*28*28, 1), )  # fc，最终Cifar10输出是10类
        #
        # self.classifier = nn.Sequential(nn.Linear(20, 10), )  # fc，最终Cifar10输出是10类
        self.flinear = nn.Linear(self.k, 128)
        self.classi = nn.Linear(128, self.k)

    def forward(self, x):
        #[batch_size,3,32,32]


        batch_size = x.size(0)
        c_part = self.class_layer(x)

        x = x.view(-1)  # 改变 x 的形状以匹配高斯层


        x = self.qfuzzy_layer(x)
        x = x.view(batch_size, -1, self.k)  # 调整输出形状为 [batch_size, 3*32*32, k]


        # x = x.permute(0,2,1)
        # print(x.shape)
        fuzzy_rule_output = torch.prod(x, dim=1)

        fuzzied_x = fuzzy_rule_output.view(batch_size, self.k)
        # print(x.shape)
        fusion_output = torch.add(c_part, self.flinear(fuzzied_x))
        # fusion_output = torch.cat((c_part, x), dim=1)

        x = self.classi(fusion_output)

        # x = c_part

        return x


from models.Dense_model import DenseNet
import torch.nn.init as init
hidden_dim = 256

class MyNetwork_scene15(nn.Module):
    """ 定义整个神经网络结构 """

    def __init__(self, class_num=15):
        super(MyNetwork_scene15, self).__init__()
        self.k = class_num

        self.class_layer = DenseNet()

        # self.fuzzy_feature = nn.Linear(200,8)
        self.qfuzzy_layer = QFuzzyLayer1(class_num) #神经网络只有1层的情况
        self.fuzzy_rule = nn.Sequential(nn.Linear(200, 1),
                                        # nn.ReLU(),
                                        # # nn.Dropout(0.5),
                                        # nn.Linear(64, 1),
                                        )  # fc，最终Cifar10输出是10类

        self.flinear = nn.Linear(self.k, hidden_dim)
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
        # [batch_size,3,32,32]
        batch_size = x.size(0)
        c_part = self.class_layer(x)

        # # x = self.fuzzy_feature(x)
        x = x.view(-1)  # 改变 x 的形状以匹配高斯层
        x = self.qfuzzy_layer(x)
        #
        x = x.view(batch_size, -1, self.k)  # 调整输出形状为 [batch_size, 200, k]
        # # print(x.shape)
        #

        # # print(x.shape)
        fuzzy_rule_output = torch.prod(x, dim=1)

        # x = x.permute(0,2,1)
        # fuzzy_rule_output = self.fuzzy_rule(x)
        x = fuzzy_rule_output.view(batch_size, self.k)

        # x = self.class_layer(x)
        # x = torch.zeros_like(fuzzy_rule_output)
        # fusion_output = torch.cat((c_part, x), dim=1)
        fusion_output = torch.add(c_part, self.flinear(x))

        # fusion_output = torch.cat((self.line1(c_part), self.line2(x) ), dim=1)
        # fusion_output = c_part
        x = self.classifier(fusion_output)
        # print(x[0][0])

        return x

from models.CNN import MyNetwork_JAFFE_DNN

hidden_dim = 512
class MyNetwork_JAFFE(nn.Module):
    """ 定义整个神经网络结构 """

    def __init__(self, class_num=15):
        super(MyNetwork_JAFFE, self).__init__()
        self.k = class_num

        self.class_layer = MyNetwork_JAFFE_DNN()

        # self.fuzzy_feature = nn.Linear(200,8)
        self.qfuzzy_layer = QFuzzyLayer(class_num)

        self.flinear = nn.Linear(self.k, hidden_dim)


        self.classi = nn.Linear(hidden_dim, self.k)

        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # [batch_size,3,32,32]
        batch_size = x.size(0)
        c_part = self.class_layer(x)

        # x = self.fuzzy_feature(x)
        x = x.view(-1)  # 改变 x 的形状以匹配高斯层
        x = self.qfuzzy_layer(x)
        #
        x = x.view(batch_size, -1, self.k)  # 调整输出形状为 [batch_size, 200, k]
        # # print(x.shape)
        # # print(x.shape)
        fuzzy_rule_output = torch.prod(x, dim=1)

        # x = x.permute(0,2,1)
        # fuzzy_rule_output = self.fuzzy_rule(x)
        fuzzied_x = fuzzy_rule_output.view(batch_size, self.k)


        fusion_output = torch.add(c_part, self.flinear(fuzzied_x))  # 只有fuzzy通过线性层 #好


        # x = self.classifier(fusion_output)
        x = self.classi(fusion_output)
        # x = self.classi(c_part)

        return x