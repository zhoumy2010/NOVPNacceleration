from models import vggnet, resnet, wrn, MLP, CNN
import torch
import torchvision.models as models
import torch.nn as nn

""" 
    生成模型列表
"""


class ModelGenerationTest(object):

    def __init__(self, args, size, num_class):
        self.args = args
        self.size = size
        self.num_class = num_class

    # 随机初始化模型参数
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def getting_models(self):
        models = []

        for i in range(self.size):
            if self.args.model == 'VGG':
                model = vggnet.VGG(16, self.num_class)
            elif self.args.model == 'res':
                if self.args.dataset == 'cifar10':
                    model = resnet.ResNet(18, self.num_class)
                    # model = CNN.CifarCNN(num_classes=self.num_class)
                elif self.args.dataset == 'imagenet':
                    model = models.resnet18()
            elif self.args.model == 'wrn':
                model = wrn.Wide_ResNet(28, 10, 0, self.num_class)
            elif self.args.model == 'mlp':
                if self.args.dataset == 'emnist':
                    model = MLP.MNIST_MLP(47)

            # 加载模型
            # model.load_state_dict(torch.load(f"./models_epoch_500_lr0.01/model_{i}.pt"))  # 加载状态字典
            # model.train()  # 确保模型处于训练模式

            # --------------1-----------------
            # 冻结模型参数，不计算梯度
            # for param in model.parameters():
            #     param.requires_grad = False
            #
            # # 添加一个全连接层
            # model.fc = nn.Linear(512, 10)
            # --------------1-----------------

            # 将模型添加到列表中
            model.apply(self.initialize_weights)
            models.append(model)

        return models

