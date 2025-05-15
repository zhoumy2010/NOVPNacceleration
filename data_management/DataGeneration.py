import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import Subset, TensorDataset, random_split, DataLoader
import torchvision
from random import Random
from math import ceil
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class Partition(object):
    """ 划分数据集 """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ 随机打乱数据集 """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]  # 数据的索引
        rng.shuffle(indexes)  # 会随机化数据索引的顺序

        # 将数据集随机均分为同等大小
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])  # 按照数据的索引随机均分打乱
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class DirichletPartitioner:
    def __init__(self, dataset, num_clients, alpha, batch_size):
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.batch_size = batch_size
        self.client_indices = self._dirichlet_partition()
        self.min_samples = min(len(indices) for indices in self.client_indices)

    def _dirichlet_partition(self):
        labels = np.array(self.dataset.targets)
        num_classes = len(set(labels))
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(self.num_clients)]

        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
            split_indices = np.split(class_indices[c], proportions)
            for i, idx in enumerate(split_indices):
                client_indices[i].extend(idx)

        return client_indices

    def get_client_data(self, client_id):
        indices = self.client_indices[client_id][:self.min_samples - (self.min_samples % self.batch_size)]
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

class DataPartitionerNoIID(object):
    """ 使用狄利克雷分布划分数据集，并确保每个客户端 batch 数量相同 """
    """ 使用狄利克雷分布划分数据集 """
    def __init__(self, data, sizes, num_classes=10, beta=0.5, batch_size=8, seed=1234):
        self.data = data
        self.partitions = []
        random.seed(seed)
        np.random.seed(seed)

        data_len = len(data)
        labels = np.array([data[i][1] for i in range(data_len)])  # 获取所有标签

        # 计算每个客户端应该有的 batch 数量
        total_batches = data_len // batch_size
        batches_per_client = total_batches // len(sizes)

        # 按类别划分数据
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        # 使用狄利克雷分布划分数据
        for _ in range(len(sizes)):
            self.partitions.append([])

        for c in range(num_classes):
            class_data = class_indices[c]
            np.random.shuffle(class_data)
            proportions = np.random.dirichlet(np.repeat(beta, len(sizes)))
            proportions = (proportions * len(class_data)).astype(int)
            proportions[-1] = len(class_data) - proportions[:-1].sum()  # 确保总和等于类别的样本数
            start = 0
            for i, prop in enumerate(proportions):
                end = start + prop
                self.partitions[i].extend(class_data[start:end])
                start = end

        for i in range(len(sizes)):
            random.shuffle(self.partitions[i])

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class CustomDataPartitioner(object):
    """按类别划分数据集"""

    def __init__(self, data, targets, node_classes):
        """
        data: 数据集
        targets: 数据集对应的标签
        node_classes: 字典，每个节点对应的类别列表
        """
        self.data = data
        self.targets = targets
        self.partitions = defaultdict(list)

        # 构造每个节点的索引
        for node, classes in node_classes.items():
            for class_label in classes:
                self.partitions[node].extend(
                    [idx for idx, label in enumerate(self.targets) if label == class_label]
                )

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class DataGeneration(object):
    """ 数据管理：为每个节点分配数据 """

    def __init__(self, args, size):
        self.args = args
        self.size = size

    def partition_dataset(self):
        print("划分数据集")

        train_loaders = []
        test_loaders = []
        class_distribution = defaultdict(lambda: defaultdict(int))

        if self.args.dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.Resize(224),  # 随机裁剪
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),  # 图像转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 对图像进行标准化
            ])

            trainset = torchvision.datasets.CIFAR10(root=self.args.datasetRoot,
                                                    train=True,
                                                    download=True,
                                                    transform=transform_train)
            """
                    测试数据集
            """
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


            partition_sizes = [1.0 / self.size for _ in range(self.size)]  # 每个元素都是 1.0 / size

            if self.args.IID == 'NoIID':
                # 使用狄利克雷分布划分数据集
                train_partitions = DataPartitionerNoIID(trainset, partition_sizes, num_classes=self.args.numClass, beta=self.args.dirichletBeta, batch_size=self.args.bs, seed=self.args.randomSeed)
                print("=====NoIID====")
            else:
                # 随机打乱数据集的索引
                train_partitions = DataPartitioner(trainset, partition_sizes)

            # 根据索引获得训练数据
            for i in range(self.size):
                train_partition = train_partitions.use(i)
                train_loader = torch.utils.data.DataLoader(train_partition,
                                                           batch_size=self.args.bs,
                                                           shuffle=True,
                                                           pin_memory=True)
                train_loaders.append(train_loader)
                # print(f"处理第 {i} 个节点的数据集")
                # 统计每个客户端的类别分布
                for _, labels in train_loader:
                    for label in labels:
                        class_distribution[i][label.item()] += 1

                testset = torchvision.datasets.CIFAR10(root=self.args.datasetRoot,
                                                       train=False,
                                                       download=True,
                                                       transform=transform_test)

                # test_partition = test_partitions.use(i)
                test_loader = torch.utils.data.DataLoader(testset,
                                                          batch_size=self.args.bs,
                                                          shuffle=True,
                                                          pin_memory=True)
                test_loaders.append(test_loader)
        elif self.args.dataset == 'mnist':
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307] * 3, std=[0.3081] * 3)
            ])

            trainset = torchvision.datasets.MNIST(root=self.args.datasetRoot,
                                                  train=True,
                                                  download=True,
                                                  transform=transform_train)

            # 测试数据集
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为3通道
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307] * 3, std=[0.3081] * 3)
            ])

            partition_sizes = [1.0 / self.size for _ in range(self.size)]  # 每个元素都是 1.0 / size

            if self.args.IID == 'NoIID':
                # 使用狄利克雷分布划分数据集
                train_partitions = DataPartitionerNoIID(trainset, partition_sizes, num_classes=10, beta=self.args.dirichletBeta, batch_size=self.args.bs, seed=self.args.randomSeed)

            else:
                # 随机打乱数据集的索引
                train_partitions = DataPartitioner(trainset, partition_sizes)

            # 根据索引获得训练数据
            for i in range(self.size):
                train_partition = train_partitions.use(i)
                train_loader = torch.utils.data.DataLoader(train_partition,
                                                           batch_size=self.args.bs,
                                                           shuffle=True,
                                                           pin_memory=True)

                train_loaders.append(train_loader)
                # 统计每个客户端的类别分布
                for _, labels in train_loader:
                    for label in labels:
                        class_distribution[i][label.item()] += 1

                testset = torchvision.datasets.MNIST(root=self.args.datasetRoot,
                                                     train=False,
                                                     download=True,
                                                     transform=transform_test)

                test_loader = torch.utils.data.DataLoader(testset,
                                                          batch_size=self.args.bs,
                                                          shuffle=True,
                                                          pin_memory=True)
                test_loaders.append(test_loader)
        elif self.args.dataset == 'cifar100':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # 先在原32x32基础上随机裁剪（带padding）
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.Resize(224),  # 统一Resize到224 (给ViT或大模型用)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 加一点颜色抖动
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],  # CIFAR100官方统计均值
                                     std=[0.2673, 0.2564, 0.2762]),  # CIFAR100官方统计标准差
            ])

            trainset = torchvision.datasets.CIFAR100(root=self.args.datasetRoot,
                                                    train=True,
                                                    download=True,
                                                    transform=transform_train)
            """
                    测试数据集
            """
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])
            ])


            partition_sizes = [1.0 / self.size for _ in range(self.size)]  # 每个元素都是 1.0 / size

            # 随机打乱数据集的索引
            # train_partitions = DataPartitioner(trainset, partition_sizes)
            if self.args.IID == 'NoIID':
                # 使用狄利克雷分布划分数据集
                train_partitions = DataPartitionerNoIID(trainset, partition_sizes, num_classes=self.args.numClass, beta=self.args.dirichletBeta, batch_size=self.args.bs, seed=self.args.randomSeed)
                print("=====NoIID====")
            else:
                # 随机打乱数据集的索引
                train_partitions = DataPartitioner(trainset, partition_sizes)

            # 根据索引获得训练数据
            for i in range(self.size):
                train_partition = train_partitions.use(i)
                train_loader = torch.utils.data.DataLoader(train_partition,
                                                           batch_size=self.args.bs,
                                                           shuffle=True,
                                                           pin_memory=True)
                train_loaders.append(train_loader)

                testset = torchvision.datasets.CIFAR100(root=self.args.datasetRoot,
                                                       train=False,
                                                       download=True,
                                                       transform=transform_test)

                # test_partition = test_partitions.use(i)
                test_loader = torch.utils.data.DataLoader(testset,
                                                          batch_size=self.args.bs,
                                                          shuffle=True,
                                                          pin_memory=True)
                test_loaders.append(test_loader)


        # 绘制热力图
        # self.plot_class_distribution(class_distribution)

        return train_loaders, test_loaders

    def plot_class_distribution(self, class_distribution):
        num_clients = len(class_distribution)
        num_classes = self.args.numClass
        heatmap_data = np.zeros((num_clients, num_classes))

        for client, dist in class_distribution.items():
            for class_id, count in dist.items():
                heatmap_data[client][class_id] = count

        # 将 heatmap_data 转换为整数类型
        heatmap_data = heatmap_data.astype(int)

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
        plt.xlabel('Class')
        plt.ylabel('Client')
        plt.title('Class Distribution per Client')

        plt.savefig(f"./picture/row_{self.args.dataset}_{self.args.randomSeed}_{self.args.size}_{self.args.IID}_{self.args.dirichletBeta}.png")
        plt.show()


def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    """
    该函数实现了基于狄利克雷分布的非独立同分布（Non-IID）数据采样。
    通过狄利克雷分布将训练数据集中的样本分配给不同的客户端，以模拟非独立同分布的数据分布。

    参数:
    y_train (np.ndarray): 训练数据集的标签数组。
    num_classes (int): 数据集中的类别数量。
    p (float): 每个客户端选择每个类别的概率。
    num_users (int): 客户端的数量。
    seed (int): 随机数种子，用于保证结果的可重复性。
    alpha_dirichlet (float, 可选): 狄利克雷分布的浓度参数，默认为100。

    返回:
    dict: 一个字典，键为客户端的编号，值为该客户端分配到的样本索引集合。
    """
    # 设置随机数种子，确保每次运行代码时生成的随机数序列相同，保证结果可复现
    np.random.seed(seed)

    # 这里将 p 重新赋值为 1，可能是代码的一个小失误，原参数 p 未被使用
    p = 1

    # 生成一个形状为 (num_users, num_classes) 的二维数组 Phi
    # 该数组使用二项分布采样，每个元素为 0 或 1，表示每个客户端是否选择了相应的类别
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))

    # 计算每个客户端选择的类别数量，对 Phi 数组按行求和
    n_classes_per_client = np.sum(Phi, axis=1)

    # 检查是否存在客户端没有选择任何类别，如果存在则重新采样
    while np.min(n_classes_per_client) == 0:
        # 找到没有选择任何类别的客户端的索引
        invalid_idx = np.where(n_classes_per_client == 0)[0]
        # 对这些客户端重新进行二项分布采样，更新 Phi 数组
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        # 重新计算每个客户端选择的类别数量
        n_classes_per_client = np.sum(Phi, axis=1)

    # 生成一个列表 Psi，长度为 num_classes
    # 列表中的每个元素是一个列表，包含选择了对应类别的客户端的索引
    Psi = [list(np.where(Phi[:, j] == 1)[0]) for j in range(num_classes)]

    # 计算每个类别被多少个客户端选择
    num_clients_per_class = np.array([len(x) for x in Psi])

    # 初始化一个空字典，用于存储每个客户端分配到的样本索引
    dict_users = {}

    # 遍历每个类别
    for class_i in range(num_classes):
        # 找到训练数据集中所有属于当前类别的样本的索引
        all_idxs = np.where(y_train == class_i)[0]

        # 从狄利克雷分布中采样，生成一个长度为 num_clients_per_class[class_i] 的概率分布
        # alpha_dirichlet 是狄利克雷分布的浓度参数，控制样本分配的均匀程度
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])

        # 根据上述概率分布，从选择了当前类别的客户端中随机选择客户端，为每个样本分配一个客户端
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        # 遍历选择了当前类别的客户端
        for client_k in Psi[class_i]:
            if client_k in dict_users:
                # 如果该客户端已经在字典中，将新分配的样本索引添加到其集合中
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
            else:
                # 如果该客户端不在字典中，为其创建一个新的集合，存储分配的样本索引
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])

    return dict_users

