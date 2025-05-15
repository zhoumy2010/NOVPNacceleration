"""
    图网络结构
"""
import os
import json
import numpy as np
import random
import scipy
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

class GraphNetwork(object):

    def __init__(self, args):
        self.args = args                   # 参数
        self.size = self.args.size                   # 节点数

        # 无向图
        if self.size == 96:
            self.fileName = '96-undirected-MITStudent_filter.txt'
            self.matrix = self.undirected_generate_adjacency()
        if self.size == 6:
            self.args.graphid = 1
            self.graph = self.select_graph()
            self.matrix = self.generate_adjacency_matrix()
        if self.size == 16:
            self.args.graphid = 5
            self.graph = self.select_graph()
            self.matrix = self.generate_adjacency_matrix()
        elif self.size == 31:
            self.fileName = '32-directed_studentRelation_filter.txt'
            self.matrix = self.directed_filtering_undirected()
        elif self.size == 284:
            filename = f'Oregon_subgraph_size_284_bian1818.npz'
            self.matrix = read_adj_matrix(filename)
        elif self.size == 290:
            filename = f'Oregon_subgraph_size_290_bian1137.npz'
            self.matrix = read_adj_matrix(filename)
        elif self.size == 291:
            filename = f'Oregon_subgraph_size_291_bian1384.npz'
            self.matrix = read_adj_matrix(filename)
        elif self.size == 468:
            filename = f'AS_subgraph_size_468_bian1542.npz'
            self.matrix = read_adj_matrix(filename)
        elif self.size == 469:
            filename = f'AS_subgraph_size_469_bian1596.npz'
            self.matrix = read_adj_matrix(filename)
        elif self.size == 506:
            filename = f'AS_subgraph_size_506_bian1647.npz'
            self.matrix = read_adj_matrix(filename)
        elif self.size == 548:
            filename = f'AS_subgraph_size_548_bian1658.npz'
            self.matrix = read_adj_matrix(filename)

        # 487 个节点
        # self.matrix = self.read_mat_file('temp20200826.mat', 'as7332')

        # 有向图(无需提取子图)  32 -> 31个节点; 167 -> 125
        # self.matrix = self.directed_filtering_undirected()

        # 无向图(无需提取子图) 96个节点 198个节点
        # self.matrix = self.undirected_generate_adjacency()

        # 提取 AS 子图10670
        filename = f"oregon_undirected_graph_matrix"
        # 保存邻接矩阵
        # save_adjacency_matrix(self.matrix, filename)
        # print(f"Matrix 已保存到 {filename}")
        # self.matrix = load_adjacency_matrix(filename)
        # self.extract_subgraphs()



        # 506(1647)  读取子图
        filename = f'Oregon_subgraph_size_291_bian748.npz'
        # self.matrix = read_adj_matrix(filename)

        # 提取最大子图：212 -> 161 个节点；241 -> 43 个节点  492 -> 487
        self.matrix = self.extract_largest_connected_subgraph(self.matrix)
        self.edges = self.count_edges(self.matrix)
        print(f"加载网络结构，边数：{self.edges}")


        # 从matlab里面获取图数据
        # self.matrix = self.read_mat_file('../temp20200826.mat', 'jazz')
        # 连通图：Infectious-409; elegans-452; PDZBase-161; facebookego1core-94; jazz-197

        # 随机获取图数据
        # self.matrix = self.random_generate_connected_graph() # 不连通 Haggle-273; as733-102

        # MATCHA 里面的图数据
        # self.matrix = self.generate_adjacency_matrix()  # 邻接矩阵

        # 从70个节点里的txt文件里面获取图数据
        # self.matrix = self.convert_adjacency_matrix()
        # self.degree = np.sum(self.matrix, axis=0)  # 节点度数

    def extract_subgraphs(self):

        matrix = np.array(self.matrix)
        min_size = 450  # 子图的最小节点数量
        max_size = 550  # 子图的最大节点数量

        # 初始化所有节点为可用状态，存储在集合中
        remaining = set(range(matrix.shape[0]))
        # 子图的编号，从 0 开始
        subgraph_id = 0

        # 只要剩余节点数量不少于最小子图节点数量，就继续循环
        while len(remaining) >= min_size:
            # 从剩余节点中随机选择一个作为种子节点
            seed = random.choice(list(remaining))
            # 从种子节点开始进行广度优先搜索，得到节点的遍历顺序和每个节点的前驱节点
            order, predecessors = breadth_first_order(matrix, i_start=seed)
            # 过滤出仍然在剩余节点集合中的节点
            bfs_nodes = [n for n in order if n in remaining]
            # 如果通过广度优先搜索得到的节点数量少于最小子图节点数量，跳出循环
            if len(bfs_nodes) < min_size:
                break
            # 确定子图的大小，在最小和最大子图节点数量之间随机选择
            size = random.randint(min_size, min(max_size, len(bfs_nodes)))
            # 从广度优先搜索得到的节点中选取前 size 个节点作为子图的节点
            nodes = bfs_nodes[:size]
            # 提取这些节点对应的子矩阵
            submatrix = matrix[nodes, :][:, nodes]

            # 提取子图，发现边数量
            matrix_fen = self.extract_largest_connected_subgraph(submatrix)
            edges = self.count_edges(matrix_fen)

            # 生成子图文件的文件名
            filename = f'Oregon_subgraph_{subgraph_id}_size_{size}_{len(matrix_fen)}_bian{edges}.npz'
            # 将子矩阵保存为 npz 文件
            # 保存邻接矩阵
            save_adj_matrix(submatrix, filename)

            # 打印保存信息
            # print(f"=======Saved subgraph {subgraph_id} with {size} nodes to {filename}_{len(matrix_fen)}_bian{edges}========")
            # 从剩余节点集合中移除已提取的节点
            for n in nodes:
                remaining.remove(n)
            # 子图编号加 1
            subgraph_id += 1

        # 打印提取完成信息和总共提取的子图数量
        print("Extraction complete. Total subgraphs:", subgraph_id)

    def undirected_generate_adjacency(self):
        edges = set()

        with open(f'./dataset/networks/{self.fileName}', 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                u = int(parts[0])
                v = int(parts[1])
                edges.add((u, v))

        # 步骤2：生成邻接矩阵
        nodes = sorted(set(u for u, v in edges).union(set(v for u, v in edges)))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        adj = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[node_to_idx[u]][node_to_idx[v]] = 1
            adj[node_to_idx[v]][node_to_idx[u]] = 1

        # 打印过滤后的邻接矩阵（可选）
        print("Filtered Adjacency Matrix:")
        # for row in adj:
        #     print(row)

        return adj

    def read_mat_file(self, file_path, variable_name):
        """
        读取 .mat 文件中指定变量的数据。

        参数：
        file_path (str): .mat 文件的路径。
        variable_name (str): 要读取的变量名。

        返回：
        variable_data: 变量的数据、无向连通图的邻接矩阵。
        """
        try:
            # 读取 .mat 文件
            mat_data = scipy.io.loadmat(file_path)

            # 获取变量数据
            variable_data = mat_data[variable_name]

            # 将稀疏矩阵转换为密集矩阵
            dense_matrix = variable_data.toarray()

            # 将密集矩阵转换为邻接矩阵的形式
            adjacency_matrix = (dense_matrix != 0).astype(int)

            return adjacency_matrix
        except Exception as e:
            print(f"读取 .mat 文件失败：{e}")
            return None

    # 计算拉普拉斯矩阵的第二小特征值及其对应的特征向量
    def eigenvalue(self, base_graph):

        # 计算拉普拉斯矩阵
        laplacian_matrix = np.diag(np.sum(base_graph, axis=1)) - base_graph
        # print(laplacian_matrix)

        # 计算拉普拉斯矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

        # 获取第二小特征值的索引
        second_smallest_index = np.argsort(eigenvalues)[1]

        # 获取第二小特征值及其对应的特征向量
        second_smallest_value = eigenvalues[second_smallest_index]

        return second_smallest_value

    def select_graph(self):
        # 预定义的基础网络拓扑结构
        # 可以通过扩展列表来添加更多内容
        Graphs = [
            # graph 0:
            # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
            [[(1, 5), (6, 7), (0, 4), (2, 3)],
             [(1, 7), (3, 6)],
             [(1, 0), (3, 7), (5, 6)],
             [(1, 2), (7, 0)],
             [(3, 1)]],

            [[(0, 1), (0, 3)],
             [(3, 1), (3, 2), (3, 4)],
             [(2, 1), (2, 4)],
             [(4, 5)]],

            # 5个节点的测试
            [[(0, 1), (0, 4)],
             [(1, 2), (1, 4)],
             [(2, 3), (3, 4)]],

            # graph 1:
            # 16-node gemetric graph as shown in Fig. A.3(a) in Appendix
            [[(4, 8), (6, 11), (7, 13), (0, 12), (5, 14), (10, 15), (2, 3), (1, 9)],
             [(11, 13), (14, 2), (5, 6), (15, 3), (10, 9)],
             [(11, 8), (2, 5), (13, 4), (14, 3), (0, 10)],
             [(11, 5), (15, 14), (13, 8)],
             [(2, 11)]],

            # graph 2:
            # 16-node gemetric graph as shown in Fig. A.3(b) in Appendix
            [[(2, 7), (12, 15), (3, 13), (5, 6), (8, 0), (9, 4), (11, 14), (1, 10)],
             [(8, 6), (0, 11), (3, 2), (5, 4), (15, 14), (1, 9)],
             [(8, 3), (0, 6), (11, 2), (4, 1), (12, 14)],
             [(8, 11), (6, 3), (0, 5)],
             [(8, 2), (0, 3), (6, 7), (11, 12)],
             [(8, 5), (6, 4), (0, 2), (11, 7)],
             [(8, 15), (3, 7), (0, 4), (6, 2)],
             [(8, 14), (5, 3), (11, 6), (0, 9)],
             [(8, 7), (15, 11), (2, 5), (4, 3), (1, 0), (13, 6)],
             [(12, 8)]],

            # graph 3:
            # 16-node gemetric graph as shown in Fig. A.3(c) in Appendix
            [[(3, 12), (4, 8), (1, 13), (5, 7), (9, 10), (11, 14), (6, 15), (0, 2)],
             [(7, 14), (2, 6), (5, 13), (8, 10), (1, 15), (0, 11), (3, 9), (4, 12)],
             [(2, 7), (3, 15), (9, 13), (6, 11), (4, 14), (10, 12), (1, 8), (0, 5)],
             [(5, 14), (1, 12), (13, 8), (9, 4), (2, 11), (7, 0)],
             [(5, 1), (14, 8), (13, 12), (10, 4), (6, 7)],
             [(5, 9), (14, 1), (13, 3), (8, 2), (11, 7)],
             [(5, 12), (14, 13), (1, 9), (8, 0)],
             [(5, 2), (14, 10), (1, 3), (9, 8), (13, 15)],
             [(5, 8), (14, 12), (1, 4), (13, 10)],
             [(5, 3), (14, 2), (9, 12), (1, 10), (13, 4)],
             [(5, 6), (14, 0), (8, 12), (1, 2)],
             [(5, 15), (9, 14)],
             [(11, 5)]],

            # graph 4:
            # 16-node erdos-renyi graph as shown in Fig 3.(b) in the main paper
            [[(2, 7), (3, 15), (13, 14), (8, 9), (1, 5), (0, 10), (6, 12), (4, 11)],
             [(12, 11), (5, 6), (14, 1), (9, 10), (15, 2), (8, 13)],
             [(12, 5), (11, 6), (1, 8), (9, 3), (2, 10)],
             [(12, 14), (11, 9), (5, 15), (0, 6), (1, 7)],
             [(12, 8), (5, 2), (11, 14), (1, 6)],
             [(12, 15), (13, 11), (10, 5), (3, 14)],
             [(12, 9)],
             [(0, 12)]],

            # graph 5, 8-node ring
            [[(0, 1), (2, 3), (4, 5), (6, 7)],
             [(0, 7), (2, 1), (4, 3), (6, 5)]]

        ]

        return Graphs[self.args.graphid] # 4-16个节点

    # 从txt文件里面获取图数据
    def convert_adjacency_matrix(self):
        # 定义文件名
        filename = self.args.netDataPath

        # 初始化邻接矩阵，假设节点ID从0开始，且最大ID不超过n-1
        # 如果不确定最大节点ID，可以先读取文件来找出最大ID
        n = self.args.size  # 假设最大的节点ID是99，因此需要100x100的矩阵
        adj_matrix = np.zeros((n, n))

        # 打开文件并读取内容
        with open(filename, 'r') as file:
            for line in file:
                # 去除行首尾的空白字符，并按空格分割
                start_node, end_node, weight = line.strip().split(' ')

                # 将字符串转换为整数
                start_node = int(start_node) - 1
                end_node = int(end_node) - 1

                # 在邻接矩阵中标记边的存在
                adj_matrix[start_node][end_node] = 1
                # 如果是无向图，还需要标记反向边
                adj_matrix[end_node][start_node] = 1

        # 输出邻接矩阵
        # print(adj_matrix)

        return adj_matrix

    def generate_adjacency_matrix(self):

        # 生成该无向图的邻接矩阵
        matrix = np.full((self.size, self.size), 0, dtype=int)

        for graph in self.graph:
            for edge in graph:
                node1, node2 = edge[0], edge[1]
                matrix[node1][node2] = 1
                matrix[node2][node1] = 1

        return matrix

    def directed_filtering_undirected(self):
        edges = set()

        with open(f'./dataset/networks/{self.fileName}', 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                u = int(parts[0])
                v = int(parts[1])
                edges.add((u, v))

        # 步骤2：生成邻接矩阵
        nodes = sorted(set(u for u, v in edges).union(set(v for u, v in edges)))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        adj = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[node_to_idx[u]][node_to_idx[v]] = 1

        # 步骤3：过滤双向连通子图
        filtered_edges = set()
        for u, v in edges:
            if (v, u) in edges:
                filtered_edges.add((u, v))
                filtered_edges.add((v, u))

        # 生成过滤后的邻接矩阵
        if filtered_edges:
            filtered_nodes = sorted(set(u for u, v in filtered_edges).union(set(v for u, v in filtered_edges)))
            filtered_node_to_idx = {node: i for i, node in enumerate(filtered_nodes)}
            filtered_n = len(filtered_nodes)
            filtered_adj = [[0] * filtered_n for _ in range(filtered_n)]
            for u, v in filtered_edges:
                filtered_adj[filtered_node_to_idx[u]][filtered_node_to_idx[v]] = 1
        else:
            filtered_adj = []

        # 打印过滤后的邻接矩阵（可选）
        print("Filtered Adjacency Matrix:")
        # for row in filtered_adj:
        #     print(row)

        return filtered_adj

    def dfs(self, adj_matrix, node, visited, subgraph_nodes):
        visited[node] = True
        subgraph_nodes.append(node)
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                self.dfs(adj_matrix, neighbor, visited, subgraph_nodes)

    def extract_largest_connected_subgraph(self, adj_matrix):
        num_nodes = len(adj_matrix)
        visited = [False] * num_nodes
        all_subgraphs = []

        # 找出所有连通子图
        for node in range(num_nodes):
            if not visited[node]:
                subgraph_nodes = []
                self.dfs(adj_matrix, node, visited, subgraph_nodes)
                all_subgraphs.append(subgraph_nodes)

        # 找到节点数最多的连通子图
        largest_subgraph = max(all_subgraphs, key=len)

        # 重新编号节点
        node_mapping = {old_node: new_node for new_node, old_node in enumerate(largest_subgraph)}

        # 构建新的邻接矩阵
        subgraph_size = len(largest_subgraph)
        subgraph_adj_matrix = [[0] * subgraph_size for _ in range(subgraph_size)]
        for i in range(subgraph_size):
            old_node_i = largest_subgraph[i]
            for j in range(subgraph_size):
                old_node_j = largest_subgraph[j]
                subgraph_adj_matrix[i][j] = adj_matrix[old_node_i][old_node_j]

        return subgraph_adj_matrix

    def count_edges(self, adj_matrix):
        num_nodes = len(adj_matrix)
        edge_count = 0
        # 遍历矩阵主对角线以上的元素
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i][j] == 1:
                    edge_count = edge_count + 1
        return edge_count


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

# 保存邻接矩阵到文件
def save_adjacency_matrix(adj, filename):
    try:
        with open(filename, 'w') as file:
            # 将邻接矩阵转换为 JSON 格式并写入文件
            json.dump(adj, file)
        print(f"邻接矩阵已成功保存到 {filename}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

# 从文件中读取邻接矩阵
def load_adjacency_matrix(filename):
    try:
        with open(filename, 'r') as file:
            # 从文件中读取 JSON 数据并转换为邻接矩阵
            adj = json.load(file)
        print(f"邻接矩阵已从 {filename} 成功读取")
        return adj
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到")
    except Exception as e:
        print(f"读取文件时出错: {e}")
    return None