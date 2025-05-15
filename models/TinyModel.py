import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个小型模型类
class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        # 使用少量的线性层和神经元
        self.fc1 = nn.Linear(5, 20)  # 输入层有5个特征，输出10个神经元
        self.fc2 = nn.Linear(20, 50)  # 输入层有5个特征，输出10个神经元
        self.fc3 = nn.Linear(50, 20)  # 输入层有5个特征，输出10个神经元
        self.fc4 = nn.Linear(20, 10)  # 输入层有5个特征，输出10个神经元
        self.fc5 = nn.Linear(10, 1)  # 输出层有1个神经元

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# 创建模型实例
# model = TinyModel()


# 随机初始化模型的参数
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)  # 使用Kaiming初始化
        nn.init.constant_(m.bias, 0)  # 将偏置初始化为0


# model.apply(initialize_weights)

# 打印模型参数数量
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total parameters: {total_params}')
#
# # 测试模型的输出
# test_input = torch.randn(1, 5)  # 生成一个随机输入
# output = model(test_input)
# print(f'Model output: {output}')
#
# for name, param in model.named_parameters():
#     print(name, param)
