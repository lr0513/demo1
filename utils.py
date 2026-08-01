import os
import random
import numpy as np
import torch
from config import cfg


def set_seed(seed: int):
    """
    固定所有随机源的种子，保证实验可复现
    """
    # 控制Python内置rando库的随机操作
    random.seed(seed)
    # 控制Numpy库的所有随即运算
    np.random.seed(seed)
    # 初始化CPU上PyTorch随机数生成器
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 单张GPU设置随机种子
        torch.cuda.manual_seed(seed)
        # 所有GPU统一设置种子
        torch.cuda.manual_seed_all(seed)
    # 强制GPU卷积使用确定性算法
    torch.backends.cudnn.deterministic = True
    # 关闭cuDNN自动寻找最优卷积算法，配合上一行共同实现完全可复现
    torch.backends.cudnn.benchmark = False


def calculate_metrics(true_labels: list, pred_labels: list, num_classes: int = cfg.model.num_classes):
    """
    手动实现多分类评估指标，不依赖sklearn，计算：准确率、宏精确率Macro-P、宏召回Macro-R、宏 F1
    :param true_labels: 真实标签列表（数字），来自验证集labels张量转list
    :param pred_labels: 预测标签列表（数字），来自torch.argmax(logits,dim=-1)得到预测id，转为list
    :param num_classes: 类别总数
    :return: 准确率、宏精确率、宏召回率、宏F1
    """
    total = len(true_labels)  # 总样本数量
    correct = 0  # 预测完全正确的样本数量

    # 初始化每个类别的tp、fp、fn
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for t, p in zip(true_labels, pred_labels):
        # 统计正确数
        if t == p:
            correct += 1
            tp[t] += 1  # 预测正确，对应类别真正例+1
        else:
            fp[p] += 1  # 预测成p类，但真实不是，假正例+1
            fn[t] += 1  # 真实是t类，但没预测出来，假负例+1

    # 准确率：所有样本里预测正确的比例，多分类通用
    accuracy = correct / total if total > 0 else 0.0

    # 计算每个类别的精确率、召回率，然后取平均（宏平均）
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(num_classes):
        # 单个类别的精确率
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        # 单个类别的召回率
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        # 单个类别的F1
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)

    # 宏平均：所有类别指标的算术平均
    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes
    macro_f1 = sum(f1_list) / num_classes

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }


def mkdir_if_not_exist(path: str):
    """文件夹不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
