import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def gaussian_membership(x, mean, std):
    """
    计算给定输入x关于均值mean、标准差std的高斯隶属度
    """
    return torch.exp(-((x - mean) ** 2) / (2 * std ** 2))


# # 实例化层,每个输入元素有3个隶属函数
# fuzzy_layer = FuzzyMembershipLayer(4, 3)
class FuzzyMembershipLayer(nn.Module):  #[batch,x]
    def __init__(self, in_features, num_memberships):
        super(FuzzyMembershipLayer, self).__init__()
        self.in_features = in_features
        self.num_memberships = num_memberships

        # 定义均值和标准差参数
        self.means = nn.Parameter(torch.randn(in_features, num_memberships))
        self.stds = nn.Parameter(torch.randn(in_features, num_memberships))

    def forward(self, x):
        batch_size = x.size(0)
        membership_values = torch.zeros(batch_size, self.in_features, self.num_memberships).to(device)

        # print(membership_values.shape)

        for i in range(self.in_features):
            for j in range(self.num_memberships):
                # print(x.shape)
                # print(gaussian_membership(x[:, i], self.means[i, j], self.stds[i, j]).shape)
                membership_values[:, i, j] = gaussian_membership(x[:, i], self.means[i, j], self.stds[i, j])

        return membership_values