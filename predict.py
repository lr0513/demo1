import json
import torch
from transformers import AutoTokenizer
from model import TextClassificationModel
from config import cfg
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_mapping():
    map_path = "./label_map.json"
    if not os.path.exists(map_path):
        raise FileNotFoundError("找不到label_map.json！请先运行训练脚本生成标签映射文件")
    with open(map_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    # 反向映射 id -> label
    id2label = {int(v): k for k, v in label2id.items()}
    return label2id, id2label


def get_model_and_tokenizer():
    """延迟加载模型、分词器、标签映射，只有调用时才加载"""
    _, id2label = load_label_mapping()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrain_name)
    model = TextClassificationModel().to(device)
    model_path = os.path.join(cfg.save.model_dir, "best_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer, id2label


def predict_single(text: str, model=None, tokenizer=None, id2label=None):
    """单条新闻标题预测"""
    # 如果没有传入模型，自动加载
    if model is None:
        model, tokenizer, id2label = get_model_and_tokenizer()

    inputs = tokenizer(
        text,
        max_length=cfg.train.max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # 注意！你的模型forward返回字典，需要取出logits
        output_dict = model(input_ids, attention_mask)
        logits = output_dict["logits"]
        pred_idx = torch.argmax(logits, dim=1).item()
    return id2label[pred_idx]


# 【只有直接运行predict.py时才启动交互，import时不会执行！】
if __name__ == "__main__":
    print("===== 今日头条新闻分类预测程序 =====")
    print("输入新闻标题进行预测，输入 exit 退出\n")
    model, tokenizer, id2label = get_model_and_tokenizer()
    while True:
        content = input("请输入新闻标题：")
        if content.strip().lower() == "exit":
            break
        result = predict_single(content, model, tokenizer, id2label)
        print(f"预测类别：{result}\n")