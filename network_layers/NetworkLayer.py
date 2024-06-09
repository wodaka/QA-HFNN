#neural network for different task
#image, time series and text
import torch
import torch.nn as nn


#简单的全连接网络提取特征
#input= [batch, input_shape]
#output = [batch, output_shape]
class classical_part_layer(nn.Module):
    """ 全连接层"""
    def __init__(self, input_feature_num, output_feature_num):
        super(classical_part_layer, self).__init__()
        self.dense_layer_1 = nn.Linear(input_feature_num, 128)
        self.dense_layer_2 = nn.Linear(128, output_feature_num)

    def forward(self, x):

        output1 =  torch.sigmoid(self.dense_layer_1(x))
        output2 = torch.sigmoid(self.dense_layer_2(output1))

        return output2

