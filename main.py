import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import ProjectConfigLoader
from utils import set_seed
from train import train
from predict import predict_single

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="实验json配置文件路径")
    args = parser.parse_args()

    # 根据传入路径加载对应实验配置
    cfg = ProjectConfigLoader(args.config_path)

    # 固定随机种子，保证每次运行结果一致
    set_seed(cfg.train.seed)

    # 启动训练流程（训练结束生成label_map.json、best_model.bin）
    train(cfg = cfg)

    # 训练完成后，才进入预测循环
    print("===== 今日头条新闻分类预测程序 =====")
    print("输入新闻标题进行预测，输入 exit 退出\n")
    while True:
        content = input("请输入新闻标题：")
        if content.strip().lower() == "exit":
            break
        result = predict_single(content, cfg)
        print(result)
        print(f"预测类别：{result}\n")
