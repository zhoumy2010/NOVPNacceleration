import os
import numpy as np
import torch
import glob


class Recorder(object):
    def __init__(self, args, id):
        # 用于存储记录指标的列表
        self.record_accuracy = list()

        self.record_timing = list()  # 记录总共消耗多少时间，本地训练时间 + 通信时间
        self.record_local_train_timing = list()  # 本地训练时间
        self.record_comm_timing = list()  # 通信时间
        self.record_comm_sum_time = list()  # 所有节点一轮内的通信时间
        self.record_losses = list()
        self.record_losses_test = list()
        self.record_trainacc = list()
        self.total_record_timing = list()

        # 存储输入参数，节点ID
        self.args = args
        self.id = id

        # 创建一个文件夹以保存记录
        self.saveFolderName = f"{self.args.size}_{self.args.dataset}_{self.args.randomSeed}_{self.args.IID}-{self.args.dirichletBeta}_{self.args.method}_batch{self.args.batchRound}_{self.args.aggRound}_lr{self.args.lr}_bs{self.args.bs}"
        if os.path.isdir(self.saveFolderName) == False and self.args.save:
            os.mkdir(self.saveFolderName)

        self.saveFolderNameAvg = f"{self.args.size}_{self.args.dataset}_{self.args.IID}-{self.args.dirichletBeta}_avg_{self.args.method}_batch{self.args.batchRound}_{self.args.aggRound}_lr{self.args.lr}_bs{self.args.bs}"
        if os.path.isdir(self.saveFolderNameAvg) == False and self.args.save:
            os.mkdir(self.saveFolderNameAvg)

    def add_new(self, local_train_time, comm_time, top1, losses, loss_test, test_acc, comm_sum_time):
        # 将新记录添加到列表中
        self.record_local_train_timing.append(local_train_time)
        self.record_comm_timing.append(comm_time)
        self.record_timing.append(local_train_time + comm_time)

        self.record_trainacc.append(top1)
        self.record_losses.append(losses)
        self.record_losses_test.append(loss_test)
        self.record_accuracy.append(test_acc)

        self.record_comm_sum_time.append(comm_sum_time)

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-acc.log', self.record_accuracy, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-losses.log', self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-losses_test.log', self.record_losses_test, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-tacc.log', self.record_trainacc, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-time.log', self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-train_timing.log', self.record_local_train_timing, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-comm_timing.log', self.record_comm_timing, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-comm_sum_timing.log', self.record_comm_sum_time, delimiter=',')
        with open(self.saveFolderName + '/ExpDescription', 'w') as f:
            f.write(str(self.args) + '\n')
            f.write(self.args.description + '\n')

    def avg_test_loss(self):
        directory = self.saveFolderName

        # 获取所有以 *losses.log 结尾的文件
        files = glob.glob(os.path.join(directory, '*losses_test.log'))

        # 存储所有文件中每行的数值
        all_lines = []

        # 遍历文件并读取每行的数值
        for file in files:
            with open(file, 'r') as f:
                lines = [float(line.strip()) for line in f if line.strip()]  # 忽略空行
                all_lines.append(lines)

        # 计算每行的平均值
        # 确保所有文件具有相同数量的行
        max_length = max(len(lines) for lines in all_lines)
        averaged_lines = []
        for i in range(max_length):
            # 提取第i行的所有数值
            line_values = [lines[i] if i < len(lines) else 0 for lines in all_lines]
            # 计算平均值
            line_average = sum(line_values) / len(files)
            averaged_lines.append(line_average)

        # 将所有行的平均值保存在新的文件中
        with open(os.path.join(directory, 'averaged_losses_test.log'), 'w') as f:
            for avg in averaged_lines:
                f.write(f"{avg}\n")

        # 将所有行的平均值保存在新的文件中
        with open(os.path.join(self.saveFolderNameAvg, f'{self.args.randomSeed}_averaged_losses_test.log'), 'w') as f:
            for avg in averaged_lines:
                f.write(f"{avg}\n")

    def avg_train_loss(self):
        directory = self.saveFolderName

        # 获取所有以 *losses.log 结尾的文件
        files = glob.glob(os.path.join(directory, '*losses.log'))

        # 存储所有文件中每行的数值
        all_lines = []

        # 遍历文件并读取每行的数值
        for file in files:
            with open(file, 'r') as f:
                lines = [float(line.strip()) for line in f if line.strip()]  # 忽略空行
                all_lines.append(lines)

        # 计算每行的平均值
        # 确保所有文件具有相同数量的行
        max_length = max(len(lines) for lines in all_lines)
        averaged_lines = []
        for i in range(max_length):
            # 提取第i行的所有数值
            line_values = [lines[i] if i < len(lines) else 0 for lines in all_lines]
            # 计算平均值
            line_average = sum(line_values) / len(files)
            averaged_lines.append(line_average)

        # 将所有行的平均值保存在新的文件中
        with open(os.path.join(directory, 'averaged_losses_train.log'), 'w') as f:
            for avg in averaged_lines:
                f.write(f"{avg}\n")

        # 将所有行的平均值保存在新的文件中
        with open(os.path.join(self.saveFolderNameAvg, f'{self.args.randomSeed}_averaged_losses_train.log'), 'w') as f:
            for avg in averaged_lines:
                f.write(f"{avg}\n")

    # def add_new(self, record_time, comp_time, comm_time, epoch_time, top1, losses, test_acc):
    #     # 将新记录添加到列表中
    #     self.total_record_timing.append(record_time)
    #     self.record_timing.append(epoch_time)
    #     self.record_comp_timing.append(comp_time)
    #     self.record_comm_timing.append(comm_time)
    #     self.record_trainacc.append(top1)
    #     self.record_losses.append(losses)
    #     self.record_accuracy.append(test_acc)
    # def save_to_file(self):
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-recordtime.log', self.total_record_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-time.log', self.record_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-comptime.log', self.record_comp_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-commtime.log', self.record_comm_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-acc.log', self.record_accuracy, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-losses.log', self.record_losses, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-tacc.log', self.record_trainacc, delimiter=',')
    #     with open(self.saveFolderName + '/ExpDescription', 'w') as f:
    #         f.write(str(self.args) + '\n')
    #         f.write(self.args.description + '\n')


