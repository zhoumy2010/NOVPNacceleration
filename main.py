import os
import random
import numpy as np
import torch
from utils import ParametersInit
from network_structure.GraphNetwork import GraphNetwork
from network_structure.GraphProcessor import GraphProcessor
from network_structure.GraphProcessorLie import GraphProcessorLie
from network_structure.GraphProcessorRow import GraphProcessorRow
from network_structure.GraphProcessorRow2 import GraphProcessorRow2
from network_structure.GraphProcessorRow3 import GraphProcessorRow3
from network_structure.GraphProcessorRow3Fast import GraphProcessorRow3Fast
from network_structure.GraphProcessorRow3FastPlus import GraphProcessorRow3FastPlus
from network_structure.GraphProcessorRow3FastCorrcoef import GraphProcessorRow3FastCorrcoef
from network_structure.GraphProcessorRow2TKD2025 import GraphProcessorRow2TKD2025
from network_structure.GraphProcessorRow2Automatica2025 import GraphProcessorRow2Automatica2025
from network_structure.GraphProcessorRow2WWW2024 import GraphProcessorRow2WWW2024
from network_structure.GraphProcessorRow2TCS2017 import GraphProcessorRow2TCS2017
from network_structure.GraphProcessorRow2IS2014 import GraphProcessorRow2IS2014
from network_structure.GraphProcessorRow2HeuristicOptimization import GraphProcessorRow2HeuristicOptimization
from network_structure.GraphProcessorRow2TKD2025Fast import GraphProcessorRow2TKD2025Fast
from network_structure.GraphProcessorRow2Automatica2025Fast import GraphProcessorRow2Automatica2025Fast
from network_structure.GraphProcessorRow2WWW2024Fast import GraphProcessorRow2WWW2024Fast
from network_structure.GraphProcessorRow2TCS2017Fast import GraphProcessorRow2TCS2017Fast
from network_structure.GraphProcessorRow2IS2014Fast import GraphProcessorRow2IS2014Fast
from network_structure.GraphProcessorRow2HeuristicOptimizationFast import GraphProcessorRow2HeuristicOptimizationFast
from data_management.DataGeneration import DataGeneration
from data_management.DataGenerationViT import DataGenerationViT
from models.ModelGeneration import ModelGeneration
from models.ModelGenerationTest import ModelGenerationTest
from simulation_training.Simulation import Simulation
from simulation_training.SimulationRow import SimulationRow
from simulation_training.SimulationRow2 import SimulationRow2
from simulation_training.SimulationRow3 import SimulationRow3
from simulation_training.SimulationRow4 import SimulationRow4
from simulation_training.SimulationRow4Test import SimulationRow4Test
from simulation_training.SimulationRow4TestError import SimulationRow4TestError
from simulation_training.SimulationRow4Testlunci import SimulationRow4Testlunci
from simulation_training.SimulationTest import SimulationTest
from simulation_training.SimulationFedAvg import SimulationFedAvg
from simulation_training.SimulationFedAvg2 import SimulationFedAvg2

if __name__ == '__main__':
    # 参数设置
    args = ParametersInit.para_init()

    torch.manual_seed(args.randomSeed)
    torch.cuda.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    random.seed(args.randomSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 分布式网络图结构
    graphNetwork = GraphNetwork(args)

    # 对图结构预处理
    if args.method == 'xuni' or args.method == 'zhenshi':
        GP = GraphProcessorRow3Fast(graphNetwork)
    elif args.method == 'xuniPlus':
        GP = GraphProcessorRow3FastPlus(graphNetwork)
    elif args.method == 'xuniCorrcoef':
        GP = GraphProcessorRow3FastCorrcoef(graphNetwork)
    elif args.method == 'TKD2025':
        GP = GraphProcessorRow2TKD2025Fast(graphNetwork)
    elif args.method == 'Automatica2025':
        GP = GraphProcessorRow2Automatica2025Fast(graphNetwork)
    elif args.method == 'WWW2024':
        GP = GraphProcessorRow2WWW2024Fast(graphNetwork)
    elif args.method == 'TCS2017':
        GP = GraphProcessorRow2TCS2017Fast(graphNetwork)
    elif args.method == 'IS2014':
        GP = GraphProcessorRow2IS2014Fast(graphNetwork)
    elif args.method == 'HO':
        GP = GraphProcessorRow2HeuristicOptimizationFast(graphNetwork)
    # 加载数据列表 返回一个训练、测试的数据列表
    train_loaders = []
    test_loaders = []

    # dataGeneration = DataGenerationViT(args, graphNetwork.size)
    # train_loaders, test_loaders = dataGeneration.partition_dataset()47.511460065841675

    # 加载模型列表
    # modelGeneration = ModelGeneration(args, args.size, args.numClass)
    # models = modelGeneration.getting_models()
    #
    # # simulator = SimulationRow4TestError(args, graphNetwork, GP, models)
    # # simulator.run()
    #
    # # 训练
    # if args.method == 'fedavg':
    #     simulator = SimulationFedAvg2(args, graphNetwork, GP, train_loaders, test_loaders, models)
    # else:
    #     simulator = SimulationRow4TestError(args, graphNetwork, GP, train_loaders, test_loaders, models)
    # simulator.run()

    # -----------------------------test---------------------------------
    # 加载模型列表
    # modelGeneration = ModelGenerationTest(args, graphNetwork.size, 10)
    # models = modelGeneration.getting_models()
    #
    # # 训练
    # simulator = SimulationTest(args, graphNetwork, GP, [], [], models)
    # simulator.run()
    # -----------------------------test-------------------------------------


