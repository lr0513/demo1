import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import cfg


def load_data(file_path: str, label_map: dict = None):
    """
    加载新闻分类数据集
    :param file_path:
    :param label_map:标签映射字典，训练集传None自动创建，验证/测试机传入训练集的label_map保持一致
    :return:
        texts: 文本列表
        labels: 数字标签列表
        label_map: 标签映射字典（仅训练集返回）
    """

    texts = []
    label_list = []  # 临时保存所有类别名，用于构建映射
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("_!_")
            if len(parts) < 4:  # 至少有4个字段（id/code/category/title），keywords可选
                continue
            label_name = parts[2]
            text = parts[3]
            texts.append(text)
            label_list.append(label_name)

    if label_map is None:
        unique_labels = sorted(list(set(label_list)))
        label_map = {name: idx for idx, name in enumerate(unique_labels)}
    labels = [label_map[name] for name in label_list]
    return texts, labels, label_map


class NewsDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer: AutoTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def get_collate_fn(self):
        """
        返回一个绑定了self.tokenizer的collate_fn
        这样collate_fn就能直接用dataset自己的tokenizer，不再依赖全局变量
        """

        def collate_fn(batch):
            """
            自定义批次组装函数，实现批次内动态padding
            职责：把DataLoader收集到的一批零散样本，加工成模型可以直接输入的张量格式
            :param batch: batch = [(text1, label1), (text2, label2), ...]
            :return: 封装好的batch字典
            """
            texts, labels = zip(*batch)

            # 关键：用 self.tokenizer，不再用全局tokenizer！
            encode = self.tokenizer(
                list(texts),
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=cfg.train.max_len
            )
            encode["labels"] = torch.tensor(labels, dtype=torch.long)
            return encode

        return collate_fn
