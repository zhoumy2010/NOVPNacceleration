import torch
import torch.nn as nn
# 定义一个简单的神经网络模型

class X2Model(nn.Module):
    def __init__(self):
        super(X2Model, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x