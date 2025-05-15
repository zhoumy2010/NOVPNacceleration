import torch
import torch.nn as nn

class FashionMnistCNN(nn.Module):
    def __init__(self):
        super(FashionMnistCNN, self).__init__()
        # 第一层卷积：输入为1个通道（灰度图像），输出为16个通道，卷积核为3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # 第二层卷积：输入为16个通道，输出为32个通道，卷积核为3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # 第三层卷积：输入为32个通道，输出为64个通道，卷积核为3x3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # 全局平均池化层，将特征图大小缩减为1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # 第一个全连接层：输入为64，输出为128
        self.fc1 = nn.Linear(64, 128)
        self.relu_fc1 = nn.ReLU()
        # 第二个全连接层：输入为128，输出为64
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU()
        # 第三个全连接层：输入为64，输出为10（Fashion-MNIST有10个分类）
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 第一层卷积+批量归一化+ReLU+池化
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        # 第二层卷积+批量归一化+ReLU+池化
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        # 第三层卷积+批量归一化+ReLU
        x = torch.relu(self.bn3(self.conv3(x)))
        # 全局平均池化
        x = self.global_avg_pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 第一个全连接层+ReLU
        x = self.relu_fc1(self.fc1(x))
        # 第二个全连接层+ReLU
        x = self.relu_fc2(self.fc2(x))
        # 第三个全连接层
        x = self.fc3(x)
        return x