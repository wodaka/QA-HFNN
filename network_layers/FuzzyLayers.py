import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from models.CNN_model import classical_layer


class denseNetwork(nn.Module):
    """ 定义整个神经网络结构 """
    def __init__(self):
        super(denseNetwork, self).__init__()

        self.linear_fuzzy = nn.Sequential(nn.Linear(1,32),
                                          nn.Linear(32,10),
                                          nn.Softmax(dim=-1),)

    def forward(self, x):
        batch_size = x.size(0)

        x_shape_1 = x.shape[1]
        x_shape_2 = x.shape[2]
        x_shape_3 = x.shape[3]

        x = x.view(x.shape[0]*x.shape[1]*x.shape[2]*x.shape[3],1)

        x = self.linear_fuzzy(x)

        x = x.view(batch_size, x_shape_1 * x_shape_2 * x_shape_3, 10)

        x = torch.mean(x, dim=1) #项数过多，乘积数字会非常小

        return x

#编码后测量
class Q_Model(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 10

            # gates with trainable parameters
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.ry2 = tq.RY(has_params=True, trainable=True)
            self.ry3 = tq.RY(has_params=True, trainable=True)
            self.ry4 = tq.RY(has_params=True, trainable=True)
            self.ry5 = tq.RY(has_params=True, trainable=True)
            self.ry6 = tq.RY(has_params=True, trainable=True)
            self.ry7 = tq.RY(has_params=True, trainable=True)
            self.ry8 = tq.RY(has_params=True, trainable=True)
            self.ry9 = tq.RY(has_params=True, trainable=True)

        def forward(self, device: tq.QuantumDevice):

            tq.cx(device, [0, 1])
            tq.cx(device, [1, 2])
            tq.cx(device, [2, 3])
            tq.cx(device, [3, 4])
            tq.cx(device, [4, 5])
            tq.cx(device, [5, 6])
            tq.cx(device, [6, 7])
            tq.cx(device, [7, 8])
            tq.cx(device, [8, 9])
            tq.cx(device, [9, 0])

            self.ry0(device, wires=0)
            self.ry1(device, wires=1)
            self.ry2(device, wires=2)
            self.ry3(device, wires=3)
            self.ry4(device, wires=4)
            self.ry5(device, wires=5)
            self.ry6(device, wires=6)
            self.ry7(device, wires=7)
            self.ry8(device, wires=8)
            self.ry9(device, wires=9)

    def __init__(self):
        super().__init__()
        self.n_wires = 10
        self.encoder = tq.GeneralEncoder([
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [0], 'func': 'ry', 'wires': [1]},
            {'input_idx': [0], 'func': 'ry', 'wires': [2]},
            {'input_idx': [0], 'func': 'ry', 'wires': [3]},
            {'input_idx': [0], 'func': 'ry', 'wires': [4]},
            {'input_idx': [0], 'func': 'ry', 'wires': [5]},
            {'input_idx': [0], 'func': 'ry', 'wires': [6]},
            {'input_idx': [0], 'func': 'ry', 'wires': [7]},
            {'input_idx': [0], 'func': 'ry', 'wires': [8]},
            {'input_idx': [0], 'func': 'ry', 'wires': [9]},
        ])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)


    def forward(self, x, use_qiskit=False):



        bsz = x.shape[0]
        x_shape_1 = x.shape[1]
        x_shape_2 = x.shape[2]
        x_shape_3 = x.shape[3]


        x = x.view(-1)


        device = tq.QuantumDevice(n_wires=10, bsz=x.shape[0], device='cuda')

        #re-upload ansaz
        self.encoder(device, x)
        self.q_layer(device)
        #
        # self.encoder(device, x)
        # self.q_layer(device)
        #
        # self.encoder(device, x)
        # self.q_layer(device)

        x = self.measure(device)

        # x = x.view(bsz,x_shape_1*x_shape_2*x_shape_3,10)
        x = x.view(bsz, 10)

        # x = torch.prod(x, dim=1)

        qout = x

        return qout




