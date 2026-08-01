import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import cfg
from utils import set_seed
from train import train

if __name__ == "__main__":
    # 第一步：固定随机种子，保证每次运行结果一致
    set_seed(cfg.train.seed)
    # 第二步：启动训练流程
    train()

