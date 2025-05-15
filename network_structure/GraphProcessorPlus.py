import time

import numpy as np
import copy
import scipy
import logging
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize
from scipy.optimize import fsolve
import sympy as sp
import networkx as nx
import random
import pickle
import math
# import pandas as pd

# 配置日志
logging.basicConfig(filename='./log/data.log', level=logging.INFO)

"""
    用于预处理通信图
"""


class GraphProcessorPlus(object):
    def __init__(self, graphNetwork):
        self.graphNetwork = graphNetwork
        self.args = graphNetwork.args

        # 记录特征值的变化
        self.list = []
        self.paths_dict_1 = {}
        self.paths_dict_2 = {}
        self.paths_1 = []

        # 记录添加的所有两条路径，以及添加之后的是否为强连通图
        self.add_paths = []
        self.add_paths_strongly_connect = []

        # 记录最大的特征值
        self.lambda_n = 0
        self.comp_time = 0

        # 保存特征值变化曲线的文件路径
        self.eigenvalue_picture_path = "./picture/row_plot_{}.png".format(self.graphNetwork.size)

        # 原始图的邻接矩阵
        self.base_graph = graphNetwork.matrix

        self.base_row_w = self.generate_row_weight_matrix(self.base_graph)

        n = len(self.base_graph)

        # 调用函数获得第二小特征值及其对应的特征向量
        matrix, self.second_bigest_val, self.second_bigest_left_vector, self.second_bigest_right_vector, self.lambda_0, self.threshold = self.row_second_bigest_eigenvalue(self.base_graph)

        # 生成的虚拟网络的第二小特征值
        self.virtual_second_bigest_val = self.second_bigest_val.copy()

        print("物理网络：")
        # print(self.base_graph)
        print(f"图结构预处理 1、计算原始网络第二大特征值：{self.second_bigest_val}，最小特征值：{self.lambda_0}")
        logging.info("%s 个节点物理网络的第二大特征值：%s", self.graphNetwork.size, self.second_bigest_val)

        if graphNetwork.args.method == 'xuniPlus':
            if self.args.matrixFlag == 0:
                # 找到所有的两跳路径
                start_time = time.time()
                # 找到所有的两跳路径
                self.paths, self.A, self.ijk_to_p, self.p_to_ijk = self.find_paths_length_2()

                # 优化添加2条边
                self.virtual_graph = self.max_independent_set()
                end_time = time.time()
                self.comp_time = end_time - start_time

                matrix = np.array(self.virtual_graph)
                # 生成子图文件的文件名
                filename = f'{self.args.size}_{self.args.method}_{self.args.randomSeed}_virtual_matrix.npz'
                # 将子矩阵保存为 npz 文件
                # 保存邻接矩阵
                save_adj_matrix(matrix, filename)
            else:
                self.virtual_graph = read_adj_matrix(
                    f'{self.args.size}_{self.args.method}_{self.args.randomSeed}_virtual_matrix.npz')


        elif graphNetwork.args.method == 'zhenshi':
            self.virtual_graph = copy.deepcopy(self.base_graph)
        else:
            self.virtual_graph = copy.deepcopy(self.base_graph)
            print("fedavg")

        print("物理网络：")
        # print(self.base_graph)
        print("虚拟网络：")
        # print(self.virtual_graph)

        # 虚拟网络每个节点的入度和出度
        self.in_degree = np.sum(self.virtual_graph, axis=0)  # 计算每列的和，即入度
        self.out_degrees = np.sum(self.virtual_graph, axis=1)  # 计算每行的和，即出度
        # print(f"每个节点出度大小：{self.in_degree}")
        # print(f"每个节点入度大小：{self.out_degrees}")

        # 获取每个节点的出度和入度节点数组
        self.virtual_outgoing_nodes, self.virtual_incoming_nodes = self.get_incoming_and_outgoing_nodes()
        print("出度节点：")
        # print(self.virtual_outgoing_nodes)
        print("入度节点：")
        # print(self.virtual_incoming_nodes)

        """
        --------- 行 ----------
        """
        # 根据邻接矩阵，生成行归一矩阵
        # self.base_row_w = self.generate_row_weight_matrix(self.base_graph)
        self.row_w = self.generate_row_weight_matrix(self.virtual_graph)
        print("真实网络的行归一权重矩阵：")
        # print(self.base_row_w)
        print("虚拟网络的行归一权重矩阵：")
        # print(self.row_w)

        # 计算特征值和右特征向量
        eigenvalues_zhenshi, right_eigenvectors_zhenshi = np.linalg.eig(self.base_row_w)
        # 提取特征值的实部
        real_eigenvalues_zhenshi = eigenvalues_zhenshi.real
        # 对特征值进行排序
        sorted_eigenvalues = np.sort(real_eigenvalues_zhenshi)[::-1]
        # print(f"真实网络的行随机矩阵的所有特征值：{sorted_eigenvalues}")

        # 计算特征值和右特征向量
        eigenvalues, right_eigenvectors = np.linalg.eig(self.row_w)
        # 提取特征值的实部
        real_eigenvalues = eigenvalues.real
        # 对特征值进行排序
        sorted_eigenvalues = np.sort(real_eigenvalues)[::-1]
        # print(f"虚拟网络的行随机矩阵的所有特征值：{sorted_eigenvalues}")

        self.row_w_n = np.linalg.matrix_power(self.row_w, 500)
        self.row_w_501 = np.linalg.matrix_power(self.row_w, 501)
        self.row_w_1000 = np.linalg.matrix_power(self.row_w, 1000)
        self.row_w_k = np.linalg.matrix_power(self.row_w, self.graphNetwork.args.aggRound)
        self.finally_weight = np.zeros(self.graphNetwork.size)

        print(f"{len(self.base_graph)}_{self.graphNetwork.args.method}耗时: {self.comp_time} 秒")

        # 生成 n 个单位列向量
        # self.row_z = self.generate_unit_vectors()
        # print("行归一的辅助向量 Z ：")
        # print(self.row_z)
        # self.test_before =[]
        # self.test_after = []
        # self.test_before = self.test_row(self.base_graph, self.base_row_w, "16_Row_record_prime_2")
        # self.test_after = self.test_row(self.virtual_graph, self.row_w, "16_Row_record_after_2")
        # self.test_draw()

    def find_stable_power(self, matrix, tolerance=1e-10, max_power=1000):
        """
        此函数用于找出矩阵 self.row_w 在多少次幂之后变成稳定值。

        参数:
        tolerance (float): 判断矩阵是否稳定的容差，默认为 1e-6。
        max_power (int): 最大迭代次数，默认为 1000。

        返回:
        int: 矩阵达到稳定值所需的幂次数，如果未达到稳定则返回 -1。
        """
        prev_matrix = matrix
        for power in range(2, max_power + 1):
            # 计算当前幂次的矩阵
            current_matrix = np.linalg.matrix_power(matrix, power)
            # 检查相邻两次幂的矩阵元素差值是否都在容差范围内
            if np.all(np.abs(current_matrix - prev_matrix) < tolerance):
                return power - 1
            prev_matrix = current_matrix
        return -1

    # 计算出列随机矩阵的第二大特征值及其对应的左右特征向量
    def row_second_bigest_eigenvalue(self, graph):

        # 计算列归一化矩阵
        row_matrix = self.generate_row_weight_matrix(graph)

        # 计算特征值和右特征向量
        eigenvalues, right_eigenvectors = np.linalg.eig(row_matrix)
        # 提取特征值的实部
        real_eigenvalues = eigenvalues.real

        # 获取特征值的索引，并找到第二大的特征值
        sorted_indices = np.argsort(real_eigenvalues)[::-1]
        second_bigest_index = sorted_indices[1]  # 第二大特征值的索引
        third_bigest_index = sorted_indices[2]

        second_bigest_eigenvalue = real_eigenvalues[second_bigest_index]
        second_bigest_right_vector = right_eigenvectors[:, second_bigest_index]

        third_virtual_value = real_eigenvalues[third_bigest_index]
        threshold = second_bigest_eigenvalue - third_virtual_value
        # print(
        #     f"第二大特征值 {second_bigest_eigenvalue} - 第三大特征值 {third_virtual_value} = {second_bigest_eigenvalue - third_virtual_value}")

        # 对于左特征向量，需要计算 (L_d^T) 的特征向量
        left_eigenvalues, left_eigenvectors = np.linalg.eig(row_matrix.T)

        second_bigest_left_index = np.argsort(left_eigenvalues)[::-1][1]
        second_bigest_left_eigenvalue = left_eigenvalues[second_bigest_left_index]
        second_bigest_left_vector = left_eigenvectors[:, second_bigest_left_index]

        smallest_index = np.argmin(real_eigenvalues)

        # 对特征值进行排序
        sorted_eigenvalues = np.sort(real_eigenvalues)[::-1]
        # print(f"列随机矩阵的所有特征值：{sorted_eigenvalues}")

        return row_matrix, second_bigest_eigenvalue, second_bigest_left_vector, second_bigest_right_vector, real_eigenvalues[smallest_index], threshold

    def is_orthogonal(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        orthogonality = abs(dot_product)
        return dot_product == 0, orthogonality

    def find_paths_length_2(self):
        """
        在无向图中查找所有长度为2的路径。
        优化版本：提前计算每个节点的邻居，减少重复遍历。
        """
        left_vector = self.second_bigest_left_vector.real
        right_vector = self.second_bigest_right_vector.real

        adj_matrix = copy.deepcopy(self.base_graph)
        adj_matrix = np.array(adj_matrix)
        # 假设 base_graph, left_vector, right_vector, base_row_w 已经定义好：
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adj_tensor = torch.from_numpy(adj_matrix)
        base_row_w_tensor = torch.from_numpy(self.base_row_w)
        left_tensor = torch.from_numpy(left_vector)
        right_tensor = torch.from_numpy(right_vector)

        base_graph = adj_tensor.to(torch.bool).cuda()  # [n, n]
        left = left_tensor.cuda()  # [n]
        right = right_tensor.cuda()  # [n]
        base_row_w = base_row_w_tensor.cuda()
        n = base_graph.shape[0]
        # 预计算常量
        norm_coef = -1.0 / (left.dot(right))  # scalar
        # 1. 计算所有 i→j→k 的可能：先把 base_graph 扩展到三维
        #    G1[i,j,k] = base_graph[i,j]，G2[i,j,k] = base_graph[j,k]
        G1 = base_graph.unsqueeze(2)  # [n, n, 1] -> broadcast->[n, n, n]
        G2 = base_graph.unsqueeze(0)  # [1, n, n]
        paths_ijk = G1 & G2  # logical and -> [n, n, n]

        # 2. 去掉已有的 i→k 直接边
        #    direct[i,k] = base_graph[i,k]，扩展成 [n,n,n]
        direct = base_graph.unsqueeze(1)  # [n,1,n] -> broadcast [n,n,n]
        # 修改后：再排除 i==k
        idxs = torch.arange(n, device=device)
        # 生成一个 [n,n,n] 的掩码，True 当且仅当 i != k
        i_not_eq_k = idxs.view(n, 1, 1) != idxs.view(1, 1, n)  # [n,n,n]

        valid = paths_ijk & (~direct) & i_not_eq_k

        # 提取有效路径索引
        paths = valid.nonzero(as_tuple=False)  # [P, 3], 每行是(i, j, k)
        P = paths.shape[0]
        # 创建索引映射
        ijk_to_p = {(i.item(), j.item(), k.item()): p for p, (i, j, k) in enumerate(paths)}
        p_to_ijk = {p: (i.item(), j.item(), k.item()) for p, (i, j, k) in enumerate(paths)}

        # 4. 提取所有 (i,j,k) 三元组的索引
        idx = valid.nonzero(as_tuple=False)  # [M, 3] 每行是 (i,j,k)

        # 5. 按公式计算 value：
        #    value = norm_coef * ( base_row_w[k,i] * left[k] * right[i]
        #                        - base_row_w[k,j] * left[k] * right[j] )
        i_idx, j_idx, k_idx = idx[:, 0], idx[:, 1], idx[:, 2]
        term1 = base_row_w[k_idx, i_idx] * left[k_idx] * right[i_idx]
        term2 = base_row_w[k_idx, j_idx] * left[k_idx] * right[j_idx]
        values = norm_coef * (term1 - term2)  # [M]

        # 6. 构造输出张量 A
        A = torch.zeros(n, n, n, device='cuda', dtype=torch.float64)
        A[i_idx, j_idx, k_idx] = values
        # 统计非零元素数量
        non_zero_count = torch.count_nonzero(A)
        print(f"非零元素数量: {non_zero_count}")

        return paths, A, ijk_to_p, p_to_ijk
    # 将 paths 保存到文件中
    def save_paths(self, paths, filename="paths.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(paths, f)
        print(f"Paths 已保存到 {filename}")

    # 从文件中加载 paths
    def load_paths(self, filename="paths.pkl"):
        with open(filename, "rb") as f:
            paths = pickle.load(f)
        print(f"Paths 已从 {filename} 加载")
        return paths

    # 将返回的稀疏矩阵存入文件中
    def save_matrix(self, matrix, filename="matrix.npz"):
        scipy.sparse.save_npz(filename, matrix)
        print(f"Matrix 已保存到 {filename}")

    # 从文件中读取稀疏矩阵
    def load_matrix(self, filename="matrix.npz"):
        matrix = scipy.sparse.load_npz(filename)
        print(f"Matrix 已从 {filename} 加载")
        return matrix

    # 转换成最大独立子集
    def convert_maximal_independent_subset(self):
        size = len(self.paths)
        print("图结构预处理 3、转换成最大独立子集；所有两跳路径size的大小：", size)

        P = self.paths.shape[0]
        device = self.paths.device

        # 展开所有组合 p1, p2
        p1_idx = torch.arange(P, device=device).unsqueeze(1).expand(P, P).reshape(-1)
        p2_idx = torch.arange(P, device=device).unsqueeze(0).expand(P, P).reshape(-1)

        a1, b1, c1 = self.paths[p1_idx][:, 0], self.paths[p1_idx][:, 1], self.paths[p1_idx][:, 2]
        a2, b2, c2 = self.paths[p2_idx][:, 0], self.paths[p2_idx][:, 1], self.paths[p2_idx][:, 2]

        # 冲突规则
        conflict1 = (a1 != a2) & (b1 == b2) & (c1 == c2)
        conflict2 = (a1 == b2) & (b1 == c2) & (c1 != a2)
        conflict3 = (a1 != c2) & (b1 == a2) & (c1 == b2)
        exception = (a1 == a2) & (b1 == b2) & (c1 != c2)

        # 最终是否冲突
        conflict = (conflict1 | conflict2 | conflict3) & (~exception)

        # 构造稀疏冲突矩阵 W ∈ [P, P]
        indices = torch.stack([p1_idx[conflict], p2_idx[conflict]], dim=0)  # [2, N]
        values = torch.ones(indices.shape[1], device=device, dtype=torch.float32)
        W = torch.sparse_coo_tensor(indices, values, size=(P, P)).coalesce()

        return W, W

    # 深度搜索有向图
    def dfs(self, graph, v, visited):
        visited[v] = True
        for i in range(len(graph)):
            if graph[v][i] == 1 and not visited[i]:
                self.dfs(graph, i, visited)

    # 判断一个有向图是否是强连通图
    def is_strongly_connected(self, adj_matrix):
        num_vertices = len(adj_matrix)

        # 从第一个顶点进行 DFS
        visited = [False] * num_vertices
        self.dfs(adj_matrix, 0, visited)

        # 检查是否所有顶点都被访问
        if not all(visited):
            return False

        # 反转图
        transposed_graph = np.transpose(adj_matrix)

        # 重置 visited
        visited = [False] * num_vertices

        # 从第一个顶点对反转图进行 DFS
        self.dfs(transposed_graph, 0, visited)

        # 检查是否所有顶点都被访问
        return all(visited)

    # 获得复数特征值的实部
    def get_real_part(self, value):
        if isinstance(value, complex):
            return value.real
        return value

    # 用贪心算法求最大独立子集问题
    def max_independent_set(self):
        print("图结构预处理 4、用贪心算法求最大独立子集问题")
        self.list.append(self.virtual_second_bigest_val)

        virtual_graph = copy.deepcopy(self.base_graph)
        # print("求最大独立子集问题前：")
        # print(virtual_graph)

        # 获取图的节点数，即邻接矩阵的行数或列数
        size = len(self.base_graph)

        # 创建一个空的集合，表示最大独立集
        independent_list = []

        count = 1

        # 保存边的数据，方便画 CaseStudy
        rows_100 = []
        rows = []
        remove_and_add_set = set()


        while True:

            # 在剩余的节点中，找到第二大特征向量差值（邻接矩阵中的非零元素数量）最大的节点
            # max_degree_node = max(remaining_nodes, key=lambda node: self.paths[node][4])

            # A 是一个三维张量 [n, n, n]
            max_index = torch.argmax(self.A)  # 一维索引
            n1, n2, n3 = self.A.shape
            # 转换为 (i, j, k)
            i = max_index // (n2 * n3)
            j = (max_index % (n2 * n3)) // n3
            k = max_index % n3

            max_degree_node = self.ijk_to_p.get((i, j, k))
            max_val = self.A[i, j, k]
            # 找不到最优的虚拟路径了
            if max_val <= 0:
                break

            # 获得 一条路径的 起始 中间 结束节点
            node1 = i
            node2 = j
            node3 = k

            v12 = virtual_graph[node1][node2]
            v23 = virtual_graph[node2][node3]
            v13 = virtual_graph[node1][node3]

            if v13 == 1:
                # print("=====这条VPN路径已经存在======")
                self.A[i, j, k] = 0
                continue

            if v12 == 0 or v23 == 0:
                # print("---------选择路径缺少边---------------")
                self.A[i, j, k] = 0
                continue

            virtual_graph[node2][node3] = 0
            virtual_graph[node1][node3] = 1

            row_matrix, virtual_value, left_vector, right_vector, l_0, threshold_new = self.row_second_bigest_eigenvalue(
                virtual_graph)

            # 判断加入的这条虚拟路径会不会使特征值变大，没有就不选这条边
            if virtual_value >= self.virtual_second_bigest_val:
                virtual_graph[node2][node3] = 1

                virtual_graph[node1][node3] = 0

                self.A[i, j, k] = 0
                # print("====特征值变大了======")
                continue
            if not self.is_strongly_connected(virtual_graph):
                virtual_graph[node2][node3] = 1

                virtual_graph[node1][node3] = 0

                self.A[i, j, k] = 0
                # print("====不是强连通图======")
                continue

            # 选择这条虚拟链路，更新特征值
            self.virtual_second_bigest_val = virtual_value

            self.list.append(self.virtual_second_bigest_val)

            # 记录添加的所有路径
            path = [count-1, node1, node2, node3, self.A[i, j, k]]
            self.add_paths.append(path)

            print(
                f"{count}  {self.virtual_second_bigest_val}  Node {max_degree_node}: {i} -> {j} -> {k} -> {self.A[i, j, k]}")
            count += 1

            # 将节点添加到独立集
            independent_list.append(max_degree_node)

            # 2) 提取所有非零路径的索引列表
            mask = self.A != 0  # [n,n,n] 布尔
            paths = mask.nonzero(as_tuple=False)  # [P,3]，每行是 (a,b,c)
            # 3) 判断与 (i,j,k) 的冲突
            #    冲突规则（与固定 (i,j,k) 对比）：
            #    1) a!=i and b==j and c==k
            #    2) a==j and b==k and c!=i
            #    3) a!=k and b==i and c==j
            #    例外： a==i and b==j and c!=k 时，不冲突
            a = paths[:, 0]
            b = paths[:, 1]
            c = paths[:, 2]

            cond1 = (a != i) & (b == j) & (c == k)
            cond2 = (a == j) & (b == k) & (c != i)
            cond3 = (a != k) & (b == i) & (c == j)
            exception = (a == i) & (b == j) & (c != k)

            conflict_mask = (cond1 | cond2 | cond3) & (~exception)

            # 4) 将所有冲突路径的 A[a,b,c] 置零
            conflicted = paths[conflict_mask]  # [M,3]
            if conflicted.numel() > 0:
                self.A[conflicted[:, 0], conflicted[:, 1], conflicted[:, 2]] = 0
            self.A[i, j, k] = 0
            if count > self.args.tau2:
                break

            # 判断是否需要更改第二大特征值
            # 获取节点的入度
            in_degrees = np.sum(virtual_graph, axis=0)  # 每列的和为节点的入度
            # 加上节点本身
            in_degrees = in_degrees + 1
            threshold = 2 * math.sqrt(2 / math.pow(in_degrees[node3], 2))
            if self.threshold <= threshold:
                print(
                    "=====================================================================更换特征向量======================================================================")
                self.threshold = threshold_new
                # 更新剩余path的值 row_matrix, virtual_value, left_vector, right_vector, l_0, threshold_new
                # 重新构造 base_row_w, left, right 等变量，并放入 GPU 上
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                left = torch.tensor(left_vector, dtype=torch.float64, device=device)
                right = torch.tensor(right_vector, dtype=torch.float64, device=device)
                base_row_w = torch.tensor(row_matrix, dtype=torch.float64, device=device)  # [n, n]

                # 更新了 矩阵 matirx之后，之前的两条路径也更新了，需要重新计算两条路径，
                # 还是 保留之前计算出来的两条路径，已经计算过一次的不用在计算，然后在重新计算所有非零元素
                # 1. 生成非零掩码
                mask = self.A != 0  # [n, n, n]，bool

                # 2. 提取所有非零元素的索引
                self.paths = mask.nonzero(as_tuple=False)  # [P, 3]，每行是一个 (i, j, k)

                # 拿出原始路径位置
                i = self.paths[:, 0]
                j = self.paths[:, 1]
                k = self.paths[:, 2]

                # 重新计算 value
                scalar = -1.0 / torch.dot(left, right)
                term1 = base_row_w[k, i] * left[k] * right[i]
                term2 = base_row_w[k, j] * left[k] * right[j]
                new_values = scalar * (term1 - term2)

                # 更新 A(i,j,k)
                self.A[i, j, k] = new_values

        print("虚拟网络：")
        # print(virtual_graph)
        print("虚拟网络的第二大特征值:", self.virtual_second_bigest_val)
        logging.info("%s 个节点虚拟网络的第二大特征值：%s", self.graphNetwork.size, self.virtual_second_bigest_val)

        # 保存特征值变化的曲线
        print("保存特征值变化的曲线")
        file_path = f"{self.graphNetwork.args.size}_{self.graphNetwork.args.randomSeed}_{self.graphNetwork.args.method}_3_plus.txt"
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for item in self.list:
                    # 将每个元素转换为字符串并写入文件，后面跟上换行符
                    file.write(str(item) + '\n')
            print(f"数据已成功保存到 {file_path}")
        except Exception as e:
            print(f"保存数据时出现错误: {e}")
        # self.plot_list_as_coordinates()

        print("记录是否为强连通图", self.add_paths_strongly_connect)
        self.virtual_graph = virtual_graph

        # n = len(self.base_graph)
        # # 遍历上三角，保证每条无向边只处理一次
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if self.base_graph[i][j]:
        #             # 将无向边(i, j)拆成(i->j)和(j->i)
        #             if not (i, j) in remove_and_add_set:
        #                 rows.append({"source": i, "target": j, "color": "grey"})
        #                 rows_100.append({"source": i, "target": j, "color": "grey"})
        #             if not (j, i) in remove_and_add_set:
        #                 rows.append({"source": j, "target": i, "color": "grey"})
        #                 rows_100.append({"source": i, "target": j, "color": "grey"})
        #  # 构造 DataFrame 并写入 Excel
        # df = pd.DataFrame(rows, columns=["source", "target", "color"])
        # df.to_excel("edges.xlsx", index=False)
        # print(f"已生成 Excel：edges.xlsx")
        #
        # # 构造 DataFrame 并写入 Excel
        # df = pd.DataFrame(rows_100, columns=["source", "target", "color"])
        # df.to_excel("edges_100.xlsx", index=False)
        # print(f"已生成 Excel：edges_100.xlsx")

        return virtual_graph

    # 生成虚拟图的邻接矩阵
    def generate_virtual_graph(self):
        print("图结构预处理 5、根据最大独立子集生成虚拟网络的邻接矩阵")
        virtual_graph = copy.deepcopy(self.base_graph)

        # 判断入度为零就不添加边
        for node in self.independent_list:
            node1 = self.paths[node][1]
            node2 = self.paths[node][2]
            node3 = self.paths[node][3]

            virtual_graph[node2][node3] = 0

            virtual_graph[node1][node3] = 1

        # # 计算拉普拉斯矩阵
        # laplacian_matrix = np.diag(np.sum(virtual_graph, axis=1)) - virtual_graph
        # # 计算拉普拉斯矩阵的特征值和特征向量
        # eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        # # 获取最大特征值的索引
        # largest_index = np.argmax(eigenvalues)
        # # 获取最大特征值
        # self.lambda_n = eigenvalues[largest_index]

        matrix, virtual_value, virtual_left_vec, virtual_right_vec, self.lambda_n, threshold = self.row_second_bigest_eigenvalue(virtual_graph)

        self.generate_xuni_directed_graph_file(virtual_graph)

        return virtual_graph

    # 根据邻接矩阵生成 .txt文件 在Matlab上测试看看虚拟图的变化
    def generate_xuni_directed_graph_file(self, adj_matrix):
        # File name to save the graph edges
        filename = "./dataset/xuni_{}_directed_graph.txt".format(self.graphNetwork.size)

        # Open the file in write mode
        with open(filename, 'w') as file:
            # Iterate over the adjacency matrix to find directed edges
            for i in range(len(adj_matrix)):
                for j in range(len(adj_matrix[i])):
                    # If there is an edge from node i to node j
                    if adj_matrix[i][j] == 1:
                        # Write the edge to the file
                        file.write(f"{i + 1} {j + 1}\n")

        return filename

    # 获取每个节点的出度和入度节点数组
    def get_incoming_and_outgoing_nodes(self):
        # 将邻接矩阵转换为NumPy数组（如果还不是）
        adj_matrix = np.array(self.virtual_graph)

        # 入度节点
        incoming_nodes = [np.where(adj_matrix[:, i] == 1)[0].tolist() for i in range(adj_matrix.shape[1])]

        # 出度节点
        outgoing_nodes = [np.where(adj_matrix[i, :] == 1)[0].tolist() for i in range(adj_matrix.shape[0])]

        return outgoing_nodes, incoming_nodes

    # 生成行归一化矩阵，入度分之一
    def generate_row_weight_matrix(self, adj_matrix):
        # 转换为 NumPy 数组
        adj_matrix = np.array(adj_matrix)

        # 获取节点的入度
        in_degrees = np.sum(adj_matrix, axis=0)  # 每列的和为节点的入度
        # 加上节点本身
        in_degrees = in_degrees + 1

        # 创建权重矩阵，初始化为零
        weight_matrix = np.zeros(adj_matrix.shape)
        incoming_nodes = [np.where(adj_matrix[:, i] == 1)[0].tolist() for i in range(adj_matrix.shape[0])]

        # 遍历每个节点，填充权重
        for i in range(adj_matrix.shape[0]):
            for node in incoming_nodes[i]:
                weight_matrix[i, node] = 1 / in_degrees[i]  # round(1 / in_degrees[i], 2)
            weight_matrix[i, i] = 1 / in_degrees[i]  # round(1 / in_degrees[i], 2)

        # col_sums = np.sum(weight_matrix, axis=0)
        # print("行归一矩阵 之前的每列元素之和:", col_sums)
        # col_sums = np.sum(weight_matrix, axis=1)
        # print("行归一矩阵 之前的每行元素之和:", col_sums)

        return weight_matrix

    def generate_unit_vectors(self):
        """
        生成n个单位列向量，每个向量的形式为 [0,..., 0, 1（在第i个位置）, 0,..., 0]T，其中i从0到n-1
        :param n: 单位向量的数量，也就是向量的维度
        :return: 包含n个单位列向量的二维数组（形状为 (n, n)）
        """
        n = self.graphNetwork.size
        unit_vectors = []
        for i in range(n):
            ei = np.zeros((n, 1))  # 创建一个全零的n行1列的列向量
            ei[i][0] = 1  # 将第i个位置的元素设置为1
            unit_vectors.append(ei)

        row_z = np.concatenate(unit_vectors, axis=1)

        # size = self.graphNetwork.size
        # adj_matrix = copy.deepcopy(self.virtual_graph)
        # weight_matrix = copy.deepcopy(self.row_w)
        # # 将邻接矩阵转换为NumPy数组（如果还不是）
        # adj_matrix = np.array(adj_matrix)
        #
        # # 入度节点
        # incoming_nodes = [np.where(adj_matrix[:, i] == 1)[0].tolist() for i in range(adj_matrix.shape[1])]
        #
        # for epoch in range(50):
        #     receiveRowZ = np.zeros((size, size))
        #     for i in range(size):
        #         for neighbor in incoming_nodes[i]:
        #             # 传输邻居节点的向量并加权
        #             receiveRowZ[i] += weight_matrix[i, neighbor] * row_z[neighbor]
        #         # 自己的向量也需要参与加权
        #         receiveRowZ[i] += weight_matrix[i, i] * row_z[i]
        #
        #     row_z = np.copy(receiveRowZ)
        #
        # # 获取row_z的行数（假设是方阵，用行数来确定循环次数）
        # num_rows = len(row_z)
        # diagonal_elements = []
        # for i in range(num_rows):
        #     diagonal_elements.append(row_z[i][i])
        # row_sum = sum(diagonal_elements)
        # print(f"行归一的辅助向量 Z 对角元素总和：{row_sum}")
        # print(diagonal_elements)
        return row_z  # 按列方向拼接所有列向量，返回最终结果

    # 判断矩阵是否为双随机混合矩阵
    def is_doubly_stochastic(self, matrix):
        row_sums = np.sum(matrix, axis=1)
        print("每行元素之和:", row_sums)

        col_sums = np.sum(matrix, axis=0)
        print("每列元素之和:", col_sums)

        total_row_sum = np.sum(row_sums)
        total_col_sum = np.sum(col_sums)

        print("行所有元素之和:", total_row_sum)
        print("列所有元素之和:", total_col_sum)

        # 检查是否所有元素非负
        if not np.all(matrix >= 0):
            negative_indices = np.where(matrix < 0)
            print("存在元素是负值！！！位于：")
            for i, j in zip(negative_indices[0], negative_indices[1]):
                print(f"行 {i}, 列 {j}：{matrix[i][j]}")
            return False

        # 检查每行元素之和是否为1
        if not np.allclose(row_sums, np.ones(matrix.shape[0])):
            return False

        # 检查每列元素之和是否为1
        if not np.allclose(col_sums, np.ones(matrix.shape[1])):
            return False

        return True

    def get_weights_matrix(self):
        print("图结构预处理：8、根据行随机矩阵列归一化")
        # 计算每列的和
        column_sums = self.W.sum(axis=0)

        col_sums = np.sum(self.W, axis=0)
        print("之前的每列元素之和:", col_sums)
        col_sums = np.sum(self.W, axis=1)
        print("之前的每行元素之和:", col_sums)

        # 进行归一化
        normalized_matrix = self.W / column_sums

        col_sums = np.sum(normalized_matrix, axis=0)
        print("-------------")
        print("归一化之后的每列元素之和:", col_sums)
        col_sums = np.sum(normalized_matrix, axis=1)
        print("归一化之后的每行元素之和:", col_sums)

        return normalized_matrix

    # 迭代计算直到收敛
    def iterate_until_convergence(self, W, V, tolerance=1e-8, max_iter=150):
        prev_V = V.copy()
        diff_norms = []

        for i in range(max_iter):
            V = W @ V

            diff_matrix = np.abs(V - prev_V)

            # 对每个列向量，判断该列是否所有元素差值都小于 tolerance
            column_converged = np.all(diff_matrix < tolerance, axis=0)  # 对每一列判断

            # 如果所有列向量的元素差值都满足条件，停止迭代
            if np.all(column_converged):
                print(f"第 {i} 次迭代收敛")
                break
            # diff_norm = np.linalg.norm(V - prev_V)
            #
            # diff_norms.append(diff_norm)
            #
            # if diff_norm < tolerance:
            #     break
            diff_norms.append(np.max(diff_matrix))  # 记录最大差值
            prev_V = V.copy()

        V_avg = V
        return V_avg, diff_norms

    # 计算差值模长
    def calculate_difference_norms(self, W, V, V_avg, max_iter=50):
        diff_norms = []
        for i in range(max_iter):
            V = W @ V
            diff_norm = np.linalg.norm(V - V_avg)
            print(f"第 {i} 轮---")
            print(f"V:{V}")
            print(f"V - V_avg:{V - V_avg}")
            print(f"diff_norm:{diff_norm}")
            diff_norms.append(diff_norm)
        print(f"V:{V}")
        return diff_norms

    def ceshi(self):
        # 初始化一个随机向量 V
        n = self.base_graph.shape[0]
        V = np.random.rand(n, 1)

        # 计算所有值的平均值
        average = np.mean(V)

        # 生成维度相同且值都为平均值的向量
        result_vector = np.full((n, 1), average)
        print(f"result_vector:{result_vector}")

        # 对W'进行迭代
        # prev_V = V.copy()
        # V_avg_W_prime, diff_norms_W_prime = self.iterate_until_convergence(self.test_w_prime, prev_V)
        # print(f"V_avg_W_prime:{V_avg_W_prime}")
        # # 对W进行迭代
        # prev_V = V.copy()
        # V_avg_W, diff_norms_W = self.iterate_until_convergence(self.W, prev_V)
        # print(f"V_avg_W:{V_avg_W}")

        # 计算W和W'的差值模长
        prev_V = V.copy()
        print(f"test_w_prime:{self.base_p}")
        print(f"prev_V:{prev_V}")
        diff_norms_W_before = self.calculate_difference_norms(self.base_p, prev_V, result_vector)

        prev_V = V.copy()
        print("-----------------------------------------")
        print(f"W:{self.p}")
        print(f"prev_V:{prev_V}")
        diff_norms_W_after = self.calculate_difference_norms(self.p, prev_V, result_vector)

        # adj_matrix = self.generate_strongly_connected_adj_matrix(n, 8)
        # print(f"random_matrix:{adj_matrix}")
        # # 计算出度拉普拉斯矩阵
        # out_degrees = np.sum(adj_matrix, axis=1)
        # # 构建出度对角矩阵 D
        # D = np.diag(out_degrees)
        # # 计算出度拉普拉斯矩阵 L_d
        # laplacian_matrix = D - adj_matrix
        # # 计算单位矩阵 I，维度与 L 相同
        # I = np.eye(laplacian_matrix.shape[0])
        # # 计算双随机混合矩阵 W
        # w = I - 0.2 * laplacian_matrix
        # prev_V = V.copy()
        # diff_norms_W_random = self.calculate_difference_norms(w, prev_V, result_vector)

        # 绘制曲线
        plt.plot(diff_norms_W_before, label='before')
        plt.plot(diff_norms_W_after, label="after")
        # plt.plot(diff_norms_W_random, label="random")
        plt.xlabel('Iterations')
        plt.ylabel('||V - V_avg||')
        plt.title('Difference Norms over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 定义收敛检查函数
    def test_check_convergence(self, vectors, tolerance=1e-5):
        # 检查所有节点的向量是否收敛到同一个向量
        reference_vector = vectors[0]
        for vector in vectors[1:]:
            if np.linalg.norm(reference_vector - vector) > tolerance:
                return False
        return True

    def test_row_version2(self, adj_matrix, weight_matrix, output_filename, max_iter=200, tolerance=1e-10):
        size = 16
        # 生成 随机向量
        acg_vector = np.full(16, 8.5)
        res = []
        vectors = np.arange(1, 17).astype(float)
        receiveRowZ = np.zeros((size, size))
        row_z = self.generate_unit_vectors()
        # receiveRowZ = np.zeros(size)
        # row_z = np.ones(size)

        print(f"-------vertors:{vectors}")

        # 计算权重矩阵的 n 次方
        matrix = np.linalg.matrix_power(weight_matrix, 1000)
        print(f"权重矩阵的 n 次方：{np.round(matrix, 2)}")

        # vectors = vectors.copy()
        for i in range(size):
            if matrix[i][i] != 0:
                # print(f"第 {i} 个元素：vectors：{vectors[i]};matrix[i][i]:{matrix[i][i]};size:{size}")
                vectors[i] = vectors[i] / (matrix[i][i])
                # print(vectors[i])

        with open(output_filename, 'w') as f:
            # 获取row_z的行数（假设是方阵，用行数来确定循环次数）
            num_rows = len(row_z)
            diagonal_elements = []
            for i in range(num_rows):
                diagonal_elements.append(row_z[i][i])

            f.write(f"Initial vectors:\nvectors:{vectors}\ndiagonal_elements:{diagonal_elements}\nweight_matrix_n:{matrix}\n\n")

        # 将邻接矩阵转换为NumPy数组（如果还不是）
        adj_matrix = np.array(adj_matrix)

        # 入度节点
        incoming_nodes = [np.where(adj_matrix[:, i] == 1)[0].tolist() for i in range(adj_matrix.shape[1])]

        for epoch in range(max_iter):
            # print(f"训练轮数： {epoch:0>5} / {max_iter}")
            new_vectors = np.zeros(size)
            receiveRowZ = np.zeros((size, size))
            # receiveRowZ = np.zeros(size)

            for i in range(size):
                for neighbor in incoming_nodes[i]:
                    # 传输邻居节点的向量并加权
                    new_vectors[i] += weight_matrix[i, neighbor] * vectors[neighbor]
                    receiveRowZ[i] += weight_matrix[i, neighbor] * row_z[neighbor]
                # 自己的向量也需要参与加权
                new_vectors[i] += weight_matrix[i, i] * vectors[i]
                receiveRowZ[i] += weight_matrix[i, i] * row_z[i]

            vectors = new_vectors.copy()
            row_z = np.copy(receiveRowZ)

            res.append(np.linalg.norm(vectors - acg_vector))

            if self.test_check_convergence(vectors, tolerance):
                print(f"第 {epoch} 收敛完成")
                print(f"new_vectors:{vectors}")
                with open(output_filename, 'a') as f:
                    f.write(f"Converged vectors at epoch {epoch}:\n{vectors}\nrow_z_sum:{row_sum}\n\n")
                return res
            with open(output_filename, 'a') as f:
                # 获取row_z的行数（假设是方阵，用行数来确定循环次数）
                num_rows = len(row_z)
                diagonal_elements = []
                for i in range(num_rows):
                    diagonal_elements.append(row_z[i][i])
                row_sum = sum(diagonal_elements)
                f.write(f"Vectors at epoch {epoch}:\nvectors:{vectors}\ndiagonal_elements:{diagonal_elements}\nrow_z_sum:{row_sum}\n\n")
        print("达到最大迭代次数，未收敛")

        return res

    def test_row(self, adj_matrix, weight_matrix, output_filename, max_iter=200, tolerance=1e-10):
        size = 16
        # 生成 随机向量
        acg_vector = np.full(16, 8.5)
        res = []
        vectors = np.arange(1, 17).astype(float)
        receiveRowZ = np.zeros((size, size))
        row_z = self.generate_unit_vectors()
        # receiveRowZ = np.zeros(size)
        # row_z = np.ones(size)

        print(f"-------vertors:{vectors}")

        # 计算权重矩阵的 n 次方
        matrix = np.linalg.matrix_power(weight_matrix, 500)
        print(f"权重矩阵的 n 次方：{np.round(matrix, 2)}")

        # vectors = vectors.copy()
        for i in range(size):
            if matrix[i][i] != 0:
                # print(f"第 {i} 个元素：vectors：{vectors[i]};matrix[i][i]:{matrix[i][i]};size:{size}")
                vectors[i] = vectors[i] / (matrix[i][i] * size)
                # print(vectors[i])

        with open(output_filename, 'w') as f:
            # 获取row_z的行数（假设是方阵，用行数来确定循环次数）
            num_rows = len(row_z)
            diagonal_elements = []
            for i in range(num_rows):
                diagonal_elements.append(row_z[i][i])

            f.write(f"Initial vectors:\nvectors:{vectors}\ndiagonal_elements:{diagonal_elements}\nweight_matrix_n:{matrix}\n\n")

        # 将邻接矩阵转换为NumPy数组（如果还不是）
        adj_matrix = np.array(adj_matrix)

        # 入度节点
        incoming_nodes = [np.where(adj_matrix[:, i] == 1)[0].tolist() for i in range(adj_matrix.shape[1])]

        for epoch in range(max_iter):
            # print(f"训练轮数： {epoch:0>5} / {max_iter}")
            new_vectors = np.zeros(size)
            receiveRowZ = np.zeros((size, size))
            # receiveRowZ = np.zeros(size)

            matrix_i = np.linalg.matrix_power(weight_matrix, epoch + 1)

            for i in range(size):
                for neighbor in incoming_nodes[i]:
                    # 传输邻居节点的向量并加权
                    new_vectors[i] += weight_matrix[i, neighbor] * vectors[neighbor]
                    receiveRowZ[i] += weight_matrix[i, neighbor] * row_z[neighbor]
                # 自己的向量也需要参与加权
                new_vectors[i] += weight_matrix[i, i] * vectors[i]
                receiveRowZ[i] += weight_matrix[i, i] * row_z[i]

            vectors = new_vectors.copy()
            row_z = np.copy(receiveRowZ)

            res.append(np.linalg.norm(vectors - acg_vector))

            if self.test_check_convergence(vectors, tolerance):
                print(f"第 {epoch} 收敛完成")
                print(f"new_vectors:{vectors}")
                with open(output_filename, 'a') as f:
                    f.write(f"Converged vectors at epoch {epoch}:\n{vectors}\nrow_z_sum:{row_sum}\n\n")
                return res
            with open(output_filename, 'a') as f:
                # 获取row_z的行数（假设是方阵，用行数来确定循环次数）
                num_rows = len(row_z)
                diagonal_elements = []
                for i in range(num_rows):
                    diagonal_elements.append(row_z[i][i])
                row_sum = sum(diagonal_elements)
                f.write(f"Vectors at epoch {epoch}:\nvectors:{vectors}\ndiagonal_elements:{diagonal_elements}\nrow_z_sum:{row_sum}\n\n")
        print("达到最大迭代次数，未收敛")

        return res

    def test_draw(self):
        # 绘制曲线
        print("before 与平均值的差值")
        print(self.test_before)
        print("after 与平均值的差值:")
        print(self.test_after)
        plt.plot(self.test_before, label='before')
        plt.plot(self.test_after, label="after")
        # plt.plot(diff_norms_W_random, label="random")
        plt.xlabel('Iterations')
        plt.ylabel('||V - V_avg||')
        plt.title('Difference Norms over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

    def sseo(self):
        print("图结构预处理 SSEO 算法 获得")

        virtual_graph = copy.deepcopy(self.base_graph)

        return virtual_graph

    # 绘制特征值变化的曲线
    def plot_list_as_coordinates(self):
        """
        绘制列表数据的折线图，其中列表的下标作为横坐标，值作为纵坐标。

        参数:
        data (list): 要绘制的数据列表。
        xlabel (str): 横坐标的标签，默认为 'Index'。
        ylabel (str): 纵坐标的标签，默认为 'Value'。
        title (str): 图形的标题，默认为 'Plotting with List Indices and Values'。
        """

        xlabel = "扰动边的数量"
        ylabel = "扰动后的特征值变化"
        title = "{} 个节点的特征值变化".format(self.graphNetwork.size)

        # 指定字体，确保支持中文
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定支持中文的字体
        plt.rcParams['axes.unicode_minus'] = False  # 确保支持负号
        plt.plot(range(len(self.list)), self.list)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # 将图形保存到文件
        plt.savefig(self.eigenvalue_picture_path)

        # plt.show()

    def generate_strongly_connected_adj_matrix(self, size, extra_edges=0):
        # 使用networkx生成一个随机连通图
        G = nx.connected_watts_strogatz_graph(size, k=2, p=0.5)

        # 将图转换为邻接矩阵
        adjacency_matrix = nx.adjacency_matrix(G).todense()

        return np.array(adjacency_matrix)

    def generate_directed_connected_graph_adjacency_matrix(self, n, extra_edges=0):
        # Step 1: 生成一个包含所有节点的有向生成树，确保基本的强连通性
        G = nx.DiGraph()
        nodes = list(range(n))
        random.shuffle(nodes)

        for i in range(1, n):
            # 从之前的节点集中随机选取一个节点，确保连通
            G.add_edge(nodes[random.randint(0, i - 1)], nodes[i])

        # Step 2: 添加一些额外的随机有向边，进一步随机化图
        possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j and not G.has_edge(i, j)]
        random.shuffle(possible_edges)

        for _ in range(min(extra_edges, len(possible_edges))):
            edge = possible_edges.pop()
            G.add_edge(edge[0], edge[1])

        # 将图转换为邻接矩阵
        adjacency_matrix = nx.adjacency_matrix(G).todense()

        return np.array(adjacency_matrix)




# 保存邻接矩阵到文件
def save_adj_matrix(adj, filename):
    try:
        # 使用 numpy 的 savez 函数保存邻接矩阵
        np.savez(filename, adj=adj)
        print(f"邻接矩阵已成功保存到 {filename}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

# 从文件中读取邻接矩阵
def read_adj_matrix(filename):
    try:
        # 使用 np.load 函数读取 .npz 文件
        data = np.load(filename)
        # 获取保存的邻接矩阵
        adj_matrix = data['adj']
        print(f"成功从 {filename} 读取邻接矩阵")
        return adj_matrix
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

