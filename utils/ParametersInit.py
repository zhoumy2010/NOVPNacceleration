"""
    参数设置
"""
import argparse

def para_init():
    # 设置参数
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    """
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100"
        ],
        default="cifar10",
    )"""
    parser.add_argument('--name', default="SGD_CIFAR10", type=str, help='experiment name')
    parser.add_argument('--description', default="SGD_CIFAR10", type=str, help='experiment description')
    parser.add_argument('--device', default="cuda", type=str, help='cuda or cpu')

    parser.add_argument('--model', default="vit", type=str, help='model name: vit/res/VGG/wrn')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=201, type=int, help='total epoch')
    parser.add_argument('--bs', default=8, type=int, help='batch size on each worker')
    parser.add_argument('--warmup', default=True, action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', default=False, action='store_true', help='use nesterov momentum or not')

    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 or fashionMnist or mnist')
    parser.add_argument('--numClass', default=10, type=int, help='class of dataset')
    parser.add_argument('--method', default='xuniCorrcoef', type=str, help='fedavg or xuni or zhenshi')
    # parser.add_argument('--methodImprove', default=0, type=int, help='是否选择改进 0 or 1')
    parser.add_argument('--IID', default='IID', type=str, help='NoIID or IID')  # gai 改 test
    parser.add_argument('--dirichletBeta', default=0.5, type=int, help='非独立同分布划分时使用的参数')
    parser.add_argument('--aggRound', default=1, type=int, help='Number of node aggregation rounds')
    parser.add_argument('--tau2', default=100000, type=int, help='Number of node aggregation rounds')

    parser.add_argument('--batchRound', default=1, type=int, help='Number of node aggregation rounds')
    parser.add_argument('--size', default=31, type=int, help='the size of base graph')
    parser.add_argument('--graphid', default=5, type=int, help='the idx of base graph 0 or 1 or 4')
    # parser.add_argument('--clipValue', default=1, type=int, help='the idx of base graph')  # gai 改
    parser.add_argument('--updateRound', default=5, type=int, help='Local node update rounds')

    parser.add_argument('--datasetRoot', default='./dataset', type=str, help='the path of dataset')
    parser.add_argument('--p', '-p', default=True, action='store_true', help='partition the dataset or not')
    # parser.add_argument('--savePath', default='./saveModel_5x_row4_16-4_zhenshi_bs8_lr0.0001_res', type=str, help='save path')
    # parser.add_argument('--netDataPath', default="dataset/socialNet_highschool.txt", type=str, help='netData path')
    parser.add_argument('--save', default=True, action='store_true', help='save medal or not')
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--randomSeed', default=1234, type=int, help='random seed')

    args = parser.parse_args()

    return args
