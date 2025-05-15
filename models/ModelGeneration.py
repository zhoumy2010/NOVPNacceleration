import copy
import os

from sklearn.neural_network import MLPClassifier

from models import vggnet, resnet, wrn, MLP, CNN, TinyModel, MnistCNN, X2Model, FashionMnistCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from vit_pytorch import ViT
""" 
    生成模型列表
"""


class ModelGeneration(object):

    def __init__(self, args, size, num_class):
        self.args = args
        self.size = size
        self.num_class = num_class

    def getting_models(self):
        models = []

        if self.args.model == 'VGG':
            model = vggnet.VGG(16, self.num_class)
        elif self.args.model == 'vit':
            # 设置 ViT 模型参数（较小的参数量，类似于 ViT-Tiny）
            image_size = 224
            patch_size = 16
            num_classes = self.num_class  # CIFAR10类别数
            dim = 128  # 较小的维度
            depth = 4  # 较浅的层数
            heads = 3  # 较少的多头数
            mlp_dim = 256  # MLP 隐藏层维度

            # 在循环内每次都新建一个模型，这样每个模型都会随机初始化参数
            # model = ViT(
            #     image_size=image_size,
            #     patch_size=patch_size,
            #     num_classes=num_classes,
            #     dim=dim,
            #     depth=depth,
            #     heads=heads,
            #     mlp_dim=mlp_dim,
            #     dropout=0.1,
            #     emb_dropout=0.1
            # )

            for i in range(self.size):
                # 在循环内每次都新建一个模型，这样每个模型都会随机初始化参数
                model = ViT(
                    image_size=image_size,
                    patch_size=patch_size,
                    num_classes=num_classes,
                    dim=dim,
                    depth=depth,
                    heads=heads,
                    mlp_dim=mlp_dim,
                    dropout=0.1,
                    emb_dropout=0.1
                )
                models.append(model)
        elif self.args.model == 'res':
            if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar3':
                model = resnet.ResNet(50, self.num_class)
                # model = CNN.CifarCNN(num_classes=self.num_class)
                # model = TinyModel.TinyModel()
            elif self.args.dataset == 'imagenet':
                model = models.resnet18()
            elif self.args.dataset == 'mnist':
                model = MnistCNN.SmallCNN()
            elif self.args.dataset == 'fashionMnist':
                model = FashionMnistCNN.FashionMnistCNN()
            elif self.args.dataset == 'x2':
                model = X2Model.X2Model()
        elif self.args.model == 'wrn':
            model = wrn.Wide_ResNet(28, 10, 0, self.num_class)
        elif self.args.model == 'mlp':
            if self.args.dataset == 'emnist':
                model = MLP.MNIST_MLP(47)
            elif self.args.dataset == 'occu':
                model = MLPClassifier(hidden_layer_sizes=(32, 32, 16),
                                      solver='adam',
                                      learning_rate='adaptive',
                                      alpha=1e-5,
                                      batch_size=8,
                                      learning_rate_init=0.01,
                                      max_iter=1000,
                                      verbose=False,
                                      early_stopping=True,
                                      tol=1e-5)


        # for i in range(self.size):
        #     models.append(copy.deepcopy(model))

        # 打印模型参数数量
        total_params = sum(p.numel() for p in models[0].parameters())
        print(f'Total parameters: {total_params}')

        # 打印模型参数
        # for name, module in models[0].named_modules():
        #     print(f"name:{name}")
        #     print(f"module:{module}---")
        # for key in models[0].state_dict():  # 遍历所有模型的参数键
        #     print(f"key :{key}")
        return models

# 定义初始化函数
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # 使用 Kaiming 正态分布初始化卷积层
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        # 初始化 BatchNorm 层
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        # 初始化全连接层
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # 如果是嵌入层 (Embedding Layer)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 使用正态分布初始化嵌入权重