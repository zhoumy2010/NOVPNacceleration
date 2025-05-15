import csv

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import time
import torch
from torch.optim import AdamW

from utils import TrainUtils, RecordUtils
import numpy as np
import copy
import math
import os
from models import vggnet, resnet, wrn, MLP, CNN, TinyModel, MnistCNN, X2Model
from tqdm import tqdm

""" 
    模拟训练
"""


class Simulation(object):

    def __init__(self, args, graphNetwork, GP, train_loaders, test_loaders, models):
        self.args = args
        self.graphNetwork = graphNetwork
        self.GP = GP
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.models = models

        self.optimizers = []
        self.optimizers_z = []
        self.criterions = []

        self.losses = []
        self.top1s = []

        self.recorder = []

        # 每个节点训练本地模型的时间、与邻居节点通信的时间
        self.local_train_time = np.zeros(graphNetwork.size)
        self.comm_time = np.zeros(graphNetwork.size)

        # 所有模型的参数列表
        self.tensor_lists = []

        # 复制的model Z
        self.models_z = []

        # 初始化 优化器、损失函数
        for i in range(self.graphNetwork.size):

            optimizer = None
            criterion = None
            if self.args.dataset == 'mnist' or self.args.dataset == 'fashionMnist':
                optimizer = optim.Adam(self.models[i].parameters(),
                                       lr=self.args.lr,
                                       weight_decay=5e-4)
                criterion = nn.CrossEntropyLoss().to(torch.device(self.args.device))
            elif self.args.dataset == 'cifar10':
                optimizer = optim.SGD(self.models[i].parameters(),
                                      lr=self.args.lr,
                                      momentum=self.args.momentum,
                                      weight_decay=5e-4,
                                      nesterov=self.args.nesterov,
                                      dampening=0)
                criterion = nn.CrossEntropyLoss().to(torch.device(self.args.device))
            elif self.args.dataset == 'cifar100':
                optimizer = AdamW(self.models[i].parameters(), self.args.lr, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss().to(torch.device(self.args.device))
            elif self.args.dataset == 'x2':
                optimizer = optim.SGD(self.models[i].parameters(),
                                      lr=self.args.lr,
                                      momentum=self.args.momentum,
                                      weight_decay=5e-4,
                                      nesterov=self.args.nesterov,
                                      dampening=0)
                criterion = nn.MSELoss().to(torch.device(self.args.device))

            self.optimizers.append(optimizer)

            # criterion = nn.CrossEntropyLoss()
            self.criterions.append(criterion)

            self.losses.append(TrainUtils.AverageMeter())
            self.top1s.append(TrainUtils.AverageMeter())

            self.recorder.append(RecordUtils.Recorder(self.args, i))

            tensor_list = list()
            for param in self.models[i].parameters():
                tensor_list.append(param.data.clone())
            self.tensor_lists.append(tensor_list)

            self.models[i].to(torch.device(self.args.device))

        # 要传给相邻节点的模型参数
        sendModel = TrainUtils.flatten_tensors(self.tensor_lists[0]).to(torch.device(self.args.device))

        # 接受相邻节点传来的模型参数
        self.receiveModels = [torch.zeros_like(sendModel) for _ in range(self.graphNetwork.size)]
        self.receiveRowZ = np.zeros((self.args.size, self.args.size))

    def run(self):
        print("开始训练")

        test_loss_history = []
        comm_sum_time = 0
        consistency_curve = []
        # 定义全局模型
        global_model = copy.deepcopy(self.models[0]).to(torch.device(self.args.device))

        epoch_average_error = []

        flag = True
        for epoch in tqdm(range(self.args.epoch), desc='Training Epochs'):
            comm_sum_time = 0
            print(f"训练轮数Row： {epoch:0>4} / {self.args.epoch}")

            # 获取所有训练数据加载器的迭代器
            iterators = [iter(loader) for loader in self.train_loaders]
            # 初始化要传输给邻居节点和接收模型的数组
            sendModel = TrainUtils.flatten_tensors(self.tensor_lists[0]).to(torch.device(self.args.device))
            self.receiveModels = [torch.zeros_like(sendModel) for _ in range(self.graphNetwork.size)]

            batch_idx = 0
            # 循环直到某个节点的训练数据耗尽
            batch_average_error = []
            savePath = f"./{self.args.size}_{self.args.dataset}_{self.args.randomSeed}_{self.args.IID}-{self.args.dirichletBeta}_{self.args.method}_batch{self.args.batchRound}_{self.args.aggRound}_lr{self.args.lr}_bs{self.args.bs}"
            if os.path.isdir(savePath) == False:
                os.mkdir(savePath)
            while True:
                batch_idx += 1
                try:
                    # 每个节点训练一个batch
                    for i in range(self.graphNetwork.size):
                        self.models[i].train()
                        self.optimizers[i].zero_grad()
                        data, target = next(iterators[i])
                        # 加载数据
                        data, target = data.to(torch.device(self.args.device)), target.to(
                            torch.device(self.args.device))

                        # 开始训练一个batch的时间
                        start_time = time.time()

                        # 前向传播
                        output = self.models[i](data)
                        loss = self.criterions[i](output, target)

                        # 记录训练损失和准确率
                        record_start = time.time()  # 记录的开始时间（记录的时间不要考虑在内）
                        acc1 = TrainUtils.comp_accuracy(output, target)
                        self.losses[i].update(loss.item(), data.size(0))
                        self.top1s[i].update(acc1[0].item(), data.size(0))
                        record_end = time.time()  # 记录的结束时间

                        # 反向传播
                        loss.backward()
                        update_learning_rate(i, self.args, self.optimizers[i], epoch, itr=batch_idx,
                                             itr_per_epoch=len(self.train_loaders[i]))

                        # 将计算出来的梯度除以常数
                        for param in self.models[i].parameters():
                            if param.grad is not None:
                                param.grad = param.grad / self.GP.row_w_n[i][i]

                        # 梯度更新
                        self.optimizers[i].step()
                        self.optimizers[i].zero_grad()

                        # 训练完一个batch的时间
                        end_time = time.time()
                        d_comp_time = (end_time - start_time - (record_end - record_start))

                        # 累加每一个batch的时间
                        self.local_train_time[i] += d_comp_time

                    # 所有节点都训练完一个batch后，进行模型聚合
                    if batch_idx % self.args.batchRound == 0:
                        # 提前计算好Fedavg的全局模型，方便记录与其模型的误差
                        models_fed = copy.deepcopy(self.models)
                        federated_average(global_model, models_fed, self.args.size)
                        global_state = global_model.state_dict()

                        comm_start_time = time.time()
                        self.aggregate_all_models_version2()
                        comm_end_time = time.time()
                        time1 = comm_end_time - comm_start_time
                        comm_sum_time += time1

                        errors = []  # 用于保存每个模型与 global_model 之间的误差
                        for model in self.models:
                            local_state = model.state_dict()
                            error = 0.0
                            # 遍历每个参数（假设各模型参数名称一致）
                            for key in global_state.keys():
                                # 计算对应参数的差异的 L2 范数平方，并累加
                                error += torch.norm(local_state[key] - global_state[key]) ** 2
                            errors.append(error.item())
                        average_error = sum(errors) / len(errors)
                        batch_average_error.append(average_error)

                    # if batch_idx % 100 == 0:
                    #     federated_average(global_model, self.models, self.args.size)
                    # else:
                    #     self.aggregate_all_models()

                except StopIteration:
                    # 当某个节点的训练数据耗尽时，停止训练
                    # self.aggregate_all_models_version2()
                    # if batch_idx % 100 == 0:
                    #     federated_average(global_model, self.models, self.args.size)
                    # else:
                    #     self.aggregate_all_models()
                    break

            # 按列求平均值
            # averaged_data = np.mean(record_train_loss, axis=0)
            # # 打开一个文件以写入模式
            # with open(f'zhenshi_fianlly4_epoch{epoch}_{self.args.aggRound}_train_loss.txt', 'w') as file:
            #     # 将数组元素转换为字符串并写入文件，每个元素占一行
            #     for element in averaged_data:
            #         file.write(str(element) + '\n')
            #
            # print(f"数组已保存到 zhenshi_fianlly4_epoch{epoch}_{self.args.aggRound}_train_loss.txt 文件中。")
            # break

            average_error_batch = sum(batch_average_error) / len(batch_average_error)
            epoch_average_error.append(average_error_batch)

            # 记录模型的数据
            val_losses = []
            for i in range(self.graphNetwork.size):
                test_acc, test_loss = TrainUtils.test(self.models[i], self.test_loaders[i], self.criterions[i])
                print(f"{epoch:0>4}  ID: %d, loss: %.3f, loss_test: %0.3f, train_acc: %.3f, test_acc: %.3f" % (i,
                                                                                                          self.losses[
                                                                                                              i].avg,
                                                                                                          test_loss,
                                                                                                          self.top1s[
                                                                                                              i].avg,
                                                                                                          test_acc))

                self.recorder[i].add_new(self.local_train_time[i], self.comm_time[i], self.top1s[i].avg, self.losses[i].avg, test_loss, test_acc, comm_sum_time)

                self.losses[i].reset()
                self.top1s[i].reset()
                self.local_train_time[i] = 0
                self.comm_time[i] = 0

                val_losses.append(test_loss)

            # 计算平均验证损失
            avg_val_loss = sum(val_losses) / len(val_losses)
            test_loss_history.append(avg_val_loss)  # 假设使用第一个模型的训练损失

            # 检查训练损失收敛 (可选)
            # if check_loss_convergence(test_loss_history):
            #     print(f"Test loss has converged at epoch {epoch}")
            #     break

            if epoch % 10 == 0:
                # 保存 errors 数组到文件
                with open(f"{savePath}/A_models_error.txt", "w") as f:
                    for error in epoch_average_error:
                        f.write(f"{error}\n")
                for i in range(self.graphNetwork.size):
                    self.recorder[i].save_to_file()
            if epoch % 100 == 0:
                self.recorder[0].avg_test_loss()
                self.recorder[0].avg_train_loss()
                for i in range(self.graphNetwork.size):
                    torch.save(self.models[i].state_dict(), f"{savePath}/model_{i}_{epoch}.pth")

        self.recorder[0].avg_test_loss()
        self.recorder[0].avg_train_loss()

        if os.path.isdir(savePath) == False:
            os.mkdir(savePath)
        for i in range(self.graphNetwork.size):
            torch.save(self.models[i].state_dict(), f"{savePath}/model_{i}_end.pth")

        # 保存 errors 数组到文件
        with open(f"{savePath}/models_error.txt", "w") as f:
            for error in epoch_average_error:
                f.write(f"{error}\n")

        # 保存一致性数据
        # save_consistency_to_file(consistency_curve, "row4_{}_{}_{}_{}_{}_lr{}_bs{}.csv".format(self.args.size, self.args.dataset, self.args.method, self.args.updateRound, self.args.aggRound, self.args.lr, self.args.bs))

    def aggregate_all_models_version2(self):

        # 初始化要传输给邻居节点和接收模型的数组
        sendModel = TrainUtils.flatten_tensors(self.tensor_lists[0]).to(torch.device(self.args.device))
        self.receiveModels = [torch.zeros_like(sendModel) for _ in range(self.graphNetwork.size)]

        # 将本地模型堆叠成参数数组
        for i in range(self.graphNetwork.size):
            # 要传给相邻节点的模型参数
            self.tensor_lists[i] = []
            for param in self.models[i].parameters():
                self.tensor_lists[i].append(param.data.clone())

        for i in range(self.graphNetwork.size):
            for node in range(self.graphNetwork.size):
                sendModel = TrainUtils.flatten_tensors(self.tensor_lists[node]).to(
                    torch.device(self.args.device))
                # 根据行随机混合矩阵中的权重加权平均
                weight = self.GP.row_w_k[i, node]
                node_weight = self.GP.row_w_n[i][node]
                # 接收邻居节点传来的模型参数
                self.receiveModels[i].add_(weight * sendModel)
            # self.receiveModels[i].div_(self.GP.finally_weight[i])

        # 更新模型
        for i in range(self.graphNetwork.size):
            # 更新模型
            with torch.no_grad():
                for new_data, param in zip(
                        TrainUtils.unflatten_tensors(  # 将一个或多个可迭代对象（例如列表、元组、字符串等）按照相同索引的元素组合成元组对
                            self.receiveModels[i].cuda(),
                            self.tensor_lists[i]),
                        self.models[i].parameters()):
                    param.data.copy_(new_data)  # 赋值


    def double_aggregate_all_models(self):
        # 聚合邻居节点模型
        epoch_convergence = 0
        flag = True
        while True:
            epoch_convergence += 1
            if epoch_convergence > self.args.aggRound:
                # print(f"====================================收敛次数大于{self.args.aggRound}轮=====================================")
                break

            # 初始化要传输给邻居节点和接收模型的数组
            sendModel = TrainUtils.flatten_tensors(self.tensor_lists[0]).to(torch.device(self.args.device))
            self.receiveModels = [torch.zeros_like(sendModel) for _ in range(self.graphNetwork.size)]
            # 将本地模型传给相邻节点
            for i in range(self.graphNetwork.size):

                # 要传给相邻节点的模型参数
                self.tensor_lists[i] = []
                for param in self.models[i].parameters():
                    self.tensor_lists[i].append(param.data.clone())

            for i in range(self.graphNetwork.size):
                for node in self.GP.virtual_incoming_nodes[i]:
                    # 将邻居节点的所有模型参数堆叠成一个张量列表
                    sendModel = TrainUtils.flatten_tensors(self.tensor_lists[node]).to(
                        torch.device(self.args.device))

                    # 根据行随机混合矩阵中的权重加权平均
                    weight = self.GP.in_degree[i] + 1

                    # 接收邻居节点传来的模型参数
                    self.receiveModels[i].add_(sendModel / weight)

            # 对传来的模型参数加权平均
            for i in range(self.graphNetwork.size):
                # 将所有模型参数堆叠成一个张量列表
                sendModel = TrainUtils.flatten_tensors(self.tensor_lists[i]).to(torch.device(self.args.device))

                # 加上自己的模型参数
                weight = self.GP.in_degree[i] + 1

                self.receiveModels[i].add_(sendModel / weight)

                # 更新模型
                with torch.no_grad():
                    for new_data, param in zip(
                            TrainUtils.unflatten_tensors(  # 将一个或多个可迭代对象（例如列表、元组、字符串等）按照相同索引的元素组合成元组对
                                self.receiveModels[i].cuda(),
                                self.tensor_lists[i]),
                            self.models[i].parameters()):
                        param.data.copy_(new_data)  # 赋值

    def aggregate_all_models(self):
        # 聚合邻居节点模型
        epoch_convergence = 0
        flag = True
        while True:
            epoch_convergence += 1
            if epoch_convergence > self.args.aggRound:
                # print(f"====================================收敛次数大于{self.args.aggRound}轮=====================================")
                break

            # 初始化要传输给邻居节点和接收模型的数组
            sendModel = TrainUtils.flatten_tensors(self.tensor_lists[0]).to(torch.device(self.args.device))
            self.receiveModels = [torch.zeros_like(sendModel) for _ in range(self.graphNetwork.size)]
            # 将本地模型传给相邻节点
            for i in range(self.graphNetwork.size):

                # 要传给相邻节点的模型参数
                self.tensor_lists[i] = []
                for param in self.models[i].parameters():
                    self.tensor_lists[i].append(param.data.clone())

            for i in range(self.graphNetwork.size):
                for node in self.GP.virtual_incoming_nodes[i]:
                    # 将邻居节点的所有模型参数堆叠成一个张量列表
                    sendModel = TrainUtils.flatten_tensors(self.tensor_lists[node]).to(
                        torch.device(self.args.device))

                    # 根据行随机混合矩阵中的权重加权平均
                    weight = self.GP.row_w[i, node]
                    node_weight = 1
                    if flag:
                        node_weight = self.GP.row_w_n[i][node] * self.args.size
                    # 接收邻居节点传来的模型参数
                    self.receiveModels[i].add_(weight * sendModel / node_weight)

            # 对传来的模型参数加权平均
            for i in range(self.graphNetwork.size):
                # 将所有模型参数堆叠成一个张量列表
                sendModel = TrainUtils.flatten_tensors(self.tensor_lists[i]).to(torch.device(self.args.device))

                # 加上自己的模型参数
                weight = self.GP.row_w[i, i]

                node_weight = 1
                if flag:
                    node_weight = self.GP.row_w_n[i][i] * self.args.size
                self.receiveModels[i].add_(weight * sendModel / node_weight)

                if epoch_convergence == self.args.aggRound:
                    self.receiveModels[i].div_(self.GP.finally_weight[i])
                # 更新模型
                with torch.no_grad():
                    for new_data, param in zip(
                            TrainUtils.unflatten_tensors(  # 将一个或多个可迭代对象（例如列表、元组、字符串等）按照相同索引的元素组合成元组对
                                self.receiveModels[i].cuda(),
                                self.tensor_lists[i]),
                            self.models[i].parameters()):
                        param.data.copy_(new_data)  # 赋值

            flag = False

        # print(f"--------------------第 {epoch_convergence} 轮次收敛--------------------------------")

# 联邦平均算法
def federated_average(global_model, client_models, size):
    """
    该函数实现联邦学习中的联邦平均算法，用于聚合多个客户端模型的参数来更新全局模型，
    并将更新后的全局模型参数同步到各个客户端模型。

    参数:
    global_model (torch.nn.Module): 全局模型，作为中心节点代表所有客户端模型的综合。
    client_models (list): 客户端模型列表，包含多个客户端本地训练后的模型。
    size (int): 客户端模型的数量，即 client_models 列表的长度。

    返回:
    无，直接在函数内部更新全局模型和客户端模型的参数。
    """
    # 提取全局模型的参数，state_dict() 返回一个字典，键是参数名，值是对应的参数张量
    global_dict = global_model.state_dict()

    # 遍历全局模型的每个参数
    for key in global_dict.keys():
        """
        对每个参数进行以下操作：
        1. 从所有客户端模型中提取该参数，使用列表推导式将这些参数组成一个列表。
        2. 使用 torch.stack 函数将列表中的张量沿着第 0 维堆叠成一个新的张量。
        3. 对堆叠后的张量沿着第 0 维求平均值，得到新的参数值。
        4. 将新的参数值更新到全局模型的参数字典中。
        """
        # 获取所有客户端模型中该参数的张量列表
        tensors = [client_models[i].state_dict()[key] for i in range(size)]
        # 检查张量类型，如果是 Long 类型，将其转换为 Float 类型
        if tensors[0].dtype == torch.long:
            tensors = [tensor.float() for tensor in tensors]
        # 堆叠张量并计算均值
        global_dict[key] = torch.stack(tensors, dim=0).mean(dim=0)

    # 将更新后的参数加载到全局模型中，完成全局模型的参数更新
    global_model.load_state_dict(global_dict)

    # 遍历所有客户端模型
    for model in client_models:
        # 将更新后的全局模型的参数加载到每个客户端模型中，实现参数同步
        model.load_state_dict(global_model.state_dict())


# 计算模型一致性（标准差均值）
def calculate_consistency(models):
    with torch.no_grad():  # 禁用梯度计算以提高效率
        param_stds = []   # 用于存储每个参数的标准差均值
        for key in models[0].state_dict():  # 遍历所有模型的参数键
            # 将每个模型中当前参数（key）的值堆叠成一个张量
            params = torch.stack([model.state_dict()[key] for model in models])
            # 计算当前参数的标准差，std(dim=0)沿模型维度计算每个元素的标准差
            std = params.std(dim=0).mean().item()  # 将标准差取均值，转换为标量
            param_stds.append(std)  # 将该参数的标准差均值加入列表
        return np.mean(param_stds)  # 返回所有参数的标准差均值

# 保存一致性数据到文件
def save_consistency_to_file(consistency_curve, filename="consistency_data.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Consistency (Mean Std)"])
        for round_idx, consistency in enumerate(consistency_curve, start=1):
            writer.writerow([round_idx, consistency])

def check_convergence(models, tolerance=1e-20):
    """
    检查多个模型是否收敛到一致。
    :param models: 包含多个模型的列表
    :param tolerance: 判断收敛的容差阈值
    :return: True/False，表示是否收敛到一致
    """
    # for i in range(1, len(models)):
    #     diff = 0.0
    #     for param1, param2 in zip(models[0].parameters(), models[i].parameters()):
    #         diff += torch.norm(param1.data - param2.data).item()
    #     if diff > tolerance:
    #         return False
    # return True

    # 提取所有模型的参数为一个列表
    model_params = [list(model.parameters()) for model in models]

    # 获取模型的参数数量
    num_models = len(model_params)

    for i in range(num_models):
        for j in range(i + 1, num_models):
            # 比较模型 i 和 j 的每个参数张量
            for param_i, param_j in zip(model_params[i], model_params[j]):
                # 计算参数之间的差值
                diff = torch.abs(param_i - param_j)
                # 检查是否所有差值都在容忍度内
                if not torch.all(diff < tolerance):
                    return False  # 如果有超出容忍度的差异，直接返回False
    return True  # 如果所有参数都在误差范围内，返回True


# 联邦平均算法
def federated_average(global_model, client_models, size):
    """
    该函数实现联邦学习中的联邦平均算法，用于聚合多个客户端模型的参数来更新全局模型，
    并将更新后的全局模型参数同步到各个客户端模型。

    参数:
    global_model (torch.nn.Module): 全局模型，作为中心节点代表所有客户端模型的综合。
    client_models (list): 客户端模型列表，包含多个客户端本地训练后的模型。
    size (int): 客户端模型的数量，即 client_models 列表的长度。

    返回:
    无，直接在函数内部更新全局模型和客户端模型的参数。
    """
    # 提取全局模型的参数，state_dict() 返回一个字典，键是参数名，值是对应的参数张量
    global_dict = global_model.state_dict()

    # 遍历全局模型的每个参数
    for key in global_dict.keys():
        """
        对每个参数进行以下操作：
        1. 从所有客户端模型中提取该参数，使用列表推导式将这些参数组成一个列表。
        2. 使用 torch.stack 函数将列表中的张量沿着第 0 维堆叠成一个新的张量。
        3. 对堆叠后的张量沿着第 0 维求平均值，得到新的参数值。
        4. 将新的参数值更新到全局模型的参数字典中。
        """
        global_dict[key] = torch.stack([client_models[i].state_dict()[key] for i in range(size)], dim=0).mean(dim=0)

    # 将更新后的参数加载到全局模型中，完成全局模型的参数更新
    global_model.load_state_dict(global_dict)

    # 遍历所有客户端模型
    for model in client_models:
        # 将更新后的全局模型的参数加载到每个客户端模型中，实现参数同步
        model.load_state_dict(global_model.state_dict())


# 联邦平均算法
def federated_average(global_model, client_models, size):
    """
    该函数实现联邦学习中的联邦平均算法，用于聚合多个客户端模型的参数来更新全局模型，
    并将更新后的全局模型参数同步到各个客户端模型。

    参数:
    global_model (torch.nn.Module): 全局模型，作为中心节点代表所有客户端模型的综合。
    client_models (list): 客户端模型列表，包含多个客户端本地训练后的模型。
    size (int): 客户端模型的数量，即 client_models 列表的长度。

    返回:
    无，直接在函数内部更新全局模型和客户端模型的参数。
    """
    # 提取全局模型的参数，state_dict() 返回一个字典，键是参数名，值是对应的参数张量
    global_dict = global_model.state_dict()

    # 遍历全局模型的每个参数
    for key in global_dict.keys():
        """
        对每个参数进行以下操作：
        1. 从所有客户端模型中提取该参数，使用列表推导式将这些参数组成一个列表。
        2. 使用 torch.stack 函数将列表中的张量沿着第 0 维堆叠成一个新的张量。
        3. 对堆叠后的张量沿着第 0 维求平均值，得到新的参数值。
        4. 将新的参数值更新到全局模型的参数字典中。
        """
        # 获取所有客户端模型中该参数的张量列表
        tensors = [client_models[i].state_dict()[key] for i in range(size)]
        # 检查张量类型，如果是 Long 类型，将其转换为 Float 类型
        if tensors[0].dtype == torch.long:
            tensors = [tensor.float() for tensor in tensors]
        # 堆叠张量并计算均值
        global_dict[key] = torch.stack(tensors, dim=0).mean(dim=0)

    # 将更新后的参数加载到全局模型中，完成全局模型的参数更新
    global_model.load_state_dict(global_dict)

    # 遍历所有客户端模型
    for model in client_models:
        # 将更新后的全局模型的参数加载到每个客户端模型中，实现参数同步
        model.load_state_dict(global_model.state_dict())

def update_learning_rate(node, args, optimizer, epoch, itr=None, itr_per_epoch=None, scale=1):
    """
    Update learning rate with linear warmup and exponential decay.

    Args:
        args (namespace): 参数对象，包含学习率等信息。
        optimizer (torch.optim.Optimizer): 优化器。
        epoch (int): 当前训练的 epoch 数。
        itr (int): 当前迭代数（可选）。
        itr_per_epoch (int): 每个 epoch 的迭代数（可选）。
        scale (int): 缩放因子（可选）。

    Notes:
        1) Linearly warmup to reference learning rate (5 epochs)
        2) Decay learning rate exponentially (epochs 30, 60, 80)
        ** note: args.lr is the reference learning rate from which to scale up
        ** note: minimum global batch-size is 256
    """
    base_lr = 0.1  # 基础学习率
    target_lr = args.lr  # 目标学习率
    total_epochs = args.epoch  # 总训练轮数
    lr_schedule = []  # 学习率衰减的阶段
    lr = None

    if args.warmup and epoch < 5:  # 如果启用了预热，并且当前 epoch 小于 5
        if target_lr <= base_lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    else:
        # 余弦退火衰减，实现更平滑的学习率调整
        lr = target_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

        # 指数衰减学习率
        # lr = target_lr
        # for e in lr_schedule:
        #     if epoch >= e:
        #         lr *= 0.1

    if lr is not None:
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # if node == 0 or node == 1:
        #     print('第 {} 轮 节点 {} Updating learning rate to {}'.format(epoch, node, lr))


# 训练损失收敛检测
def check_loss_convergence(loss_history, window_size=10, threshold=1e-4):
    """
    检查训练损失是否已经收敛

    Args:
        loss_history (list): 历史损失值列表
        window_size (int): 用于计算损失变化的窗口大小
        threshold (float): 收敛阈值，当损失变化小于此值时认为收敛

    Returns:
        bool: 是否收敛
    """
    if len(loss_history) < 2 * window_size:
        return False

    # 计算前一个窗口和当前窗口的平均损失
    prev_window = loss_history[-2 * window_size:-window_size]
    current_window = loss_history[-window_size:]

    prev_avg = sum(prev_window) / window_size
    curr_avg = sum(current_window) / window_size

    # 计算损失变化率
    loss_change = abs(prev_avg - curr_avg) / prev_avg if prev_avg != 0 else abs(curr_avg)

    return loss_change < threshold