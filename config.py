# 读取json配置文件，封装成python对象
import json
from dataclasses import dataclass


@dataclass
class DataConfig:
    train_path: str
    dev_path: str
    test_path: str


@dataclass
class TrainConfig:
    seed: int
    max_len: int
    batch_size: int
    lr: float
    epoch: int
    grad_clip_norm: float
    early_stop_patience: int


@dataclass
class ModelConfig:
    """模型结构配置"""
    pretrain_name: str
    dropout: float
    num_classes: int


@dataclass
class SaveConfig:
    """保存路径配置"""
    model_dir: str
    log_dir: str


# 全局配置总类，一次性读取所有配置
class Config:
    def __init__(self, json_path: str = "config.json"):
        with open(json_path) as f:
            config = json.load(f)

        self.data = DataConfig(**config["data"])
        self.train = TrainConfig(**config["train"])
        self.model = ModelConfig(**config["model"])
        self.save = SaveConfig(**config["save"])

cfg = Config()