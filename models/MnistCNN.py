import torch
import torch.nn as nn

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv = nn.Sequential(
            # [BATCH_SIZE, 1, 28, 28]
            nn.Conv2d(1, 32, 5, 1, 2),
            # [BATCH_SIZE, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [BATCH_SIZE, 32, 14, 14]
            nn.Conv2d(32, 64, 5, 1, 2),
            # [BATCH_SIZE, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)
            # [BATCH_SIZE, 64, 7, 7]
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


# 定义一个小型卷积神经网络模型
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        # 第一层卷积：输入为1个通道（灰度图像），输出为8个通道，卷积核为3x3
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # 第二层卷积：输入为8个通道，输出为16个通道，卷积核为3x3
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # 全局平均池化层，将特征图大小缩减为1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层：输入为16，输出为10（MNIST有10个分类）
        self.fc1 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 第一层卷积+ReLU
        x = torch.max_pool2d(x, 2)  # 池化层，将图像尺寸减半
        x = torch.relu(self.conv2(x))  # 第二层卷积+ReLU
        x = torch.max_pool2d(x, 2)  # 再次池化
        x = self.global_avg_pool(x)  # 全局平均池化
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)  # 全连接层
        return x


# 创建模型实例
# model = SmallCNN()
# # 打印模型参数数量
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total parameters: {total_params}')
