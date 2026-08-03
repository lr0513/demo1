import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import cfg
from utils import set_seed
from train import train
from predict import predict_single

if __name__ == "__main__":
    # 第一步：固定随机种子，保证每次运行结果一致
    set_seed(cfg.train.seed)
    # 第二步：启动训练流程（训练结束生成label_map.json、best_model.bin）
    train()

    # 训练完成后，才进入预测循环
    print("===== 今日头条新闻分类预测程序 =====")
    print("输入新闻标题进行预测，输入 exit 退出\n")
    while True:
        content = input("请输入新闻标题：")
        if content.strip().lower() == "exit":
            break
        result = predict_single(content)
        print(f"预测类别：{result}\n")