import torch.nn as nn
from transformers import AutoModel


class TextClassificationModel(nn.Module):
    """
    文本分类模型
    结构：预训练语言模型 + Drought + 全连接分类头
    """

    def __init__(self, pretrain_name: str, dropout: float, hidden_size: int, num_classes: int):
        super(TextClassificationModel, self).__init__()
        # 加载预训练语言模型
        self.model = AutoModel.from_pretrained(pretrain_name)
        # Dropout层：防止过拟合，随即损失一部分神经元使其置0
        self.dropout = nn.Dropout(dropout)
        # 分类全连接层：把768维的向量映射到类别数
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_idx, attention_mask, labels=None):
        """
        前向传播函数
        :param input_idx: [batch_size, seq_len] tokenizer输出，文本转换后的数字编号，包含[CLS]、[SEP]和文字 id。
        :param attention_mask: [batch_size, seq_len] 注意力掩码，标记哪些是真实token哪些是padding
        :param labels: [batch_size] 真实标签，训练时传入用来算loss，推理时不传
        :return: 返回包含 loss、logits 的字典
        """
        # 预训练模型输出，last_hidden_state是最后一层的所有token向量
        outputs = self.model(input_ids=input_idx, attention_mask=attention_mask)
        # 取第0个token（CLS）的向量作为整个句子的表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, token位置, 向量维度]
        # 过dropout，训练时随机失活，推理时自动失效
        cls_embedding = self.dropout(cls_embedding)
        # 全连接层得到logits，模型输出的未经过归一化的原始类别分值
        logits = self.classifier(cls_embedding)

        # 如果传入的标签，就计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # 统一返回字典，方便取值
        return {
            "loss": loss,
            "logits": logits
        }
