import json

import swanlab
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from config import cfg
from dataset import load_data, NewsDataset, collate_fn, tokenizer
from model import TextClassificationModel
from evaluate import evaluate
from utils import mkdir_if_not_exist


def train():
    # ========== 1. 基础环境准备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    mkdir_if_not_exist(cfg.save.model_dir)
    mkdir_if_not_exist(cfg.save.log_dir)

    swanlab.init(
        project=cfg.project.project,
        experiment_name=cfg.project.experiment_name,
        config={
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
            "data": cfg.data.__dict__,
            "device": str(device)
        }
    )

    # ========== 2. 加载数据集 ==========
    # 先加载训练集，自动生成标签映射
    train_texts, train_labels, label_map = load_data(cfg.data.train_path)
    # 验证集和测试集必须用训练集的标签映射，保证类别对应一致
    dev_texts, dev_labels, _ = load_data(cfg.data.dev_path, label_map=label_map)
    test_texts, test_labels, _ = load_data(cfg.data.test_path, label_map=label_map)

    # 打印类别对应关系，方便后续对照
    print("类别映射关系：", label_map)

    swanlab.config["label_map"] = label_map

    # 构建数据集对象
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    dev_dataset = NewsDataset(dev_texts, dev_labels, tokenizer)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer)

    # 构建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"数据集加载完成：训练集{len(train_dataset)}条，验证集{len(dev_dataset)}条，测试集{len(test_dataset)}条")

    # ========== 3. 初始化模型、优化器、学习率调度器 ==========
    model = TextClassificationModel().to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr)

    total_steps = len(train_loader) * cfg.train.epoch
    scheduler = get_linear_schedule_with_warmup(  # 学习率调度器
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 预热步数：总步数的10%
        num_training_steps=total_steps  # 完整训练总步数
    )

    # ========== 4. 早停与最优模型相关变量 ==========
    best_dev_acc = 0.0  # 记录验证集历史最高准确率
    early_stop_counter = 0  # 连续多少轮验证集没有提升
    best_model_path = f"{cfg.save.model_dir}/best_model.bin"

    # ========== 5. 训练循环 ==========
    print("开始训练...")
    for epoch in range(cfg.train.epoch):
        model.train()
        epoch_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip_norm)

            optimizer.step()
            scheduler.step()  # step计数+1，计算、更新下一轮lr

            epoch_train_loss += loss.item()

        # ========== 6. 每轮结束，验证集评估 ==========
        avg_train_loss = epoch_train_loss / len(train_loader)
        dev_metrics = evaluate(model, dev_loader, device)

        print(f"Epoch {epoch + 1}/{cfg.train.epoch}")
        print(
            f"训练集损失: {avg_train_loss:.4f} | 验证集准确率: {dev_metrics['accuracy']:.4f} | 验证集F1: {dev_metrics['f1']:.4f}")

        # 记录每轮指标，自动生成曲线
        swanlab.log({
            "train_loss": avg_train_loss,
            "val_accuracy": dev_metrics["accuracy"],
            "val_precision": dev_metrics["precision"],
            "val_recall": dev_metrics["recall"],
            "val_f1": dev_metrics["f1"],
            "learning_rate": scheduler.get_last_lr()[0]
        }, step=epoch + 1)

        # ========== 7. 保存最优模型 + 早停判断 ==========
        if dev_metrics["accuracy"] > best_dev_acc:
            best_dev_acc = dev_metrics["accuracy"]
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 验证集准确率刷新！保存最优模型，当前最佳准确率: {best_dev_acc:.4f}")
            swanlab.config["best_val_accuracy"] = best_dev_acc
        else:
            early_stop_counter += 1
            print(f"⏳ 验证集未提升，连续{early_stop_counter}轮，最佳准确率: {best_dev_acc:.4f}")

            if early_stop_counter >= cfg.train.early_stop_patience:
                print(f"\n🛑 触发早停！连续{cfg.train.early_stop_patience}轮验证集无提升，停止训练")
                break
        print("-" * 60)

    # ========== 8. 训练结束，加载最优模型跑测试集 ==========
    print("\n训练结束，加载最优模型进行测试集评估...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 50)
    print("📊 测试集最终结果（基于验证集最优模型）")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print(f"宏精确率: {test_metrics['precision']:.4f}")
    print(f"宏召回率: {test_metrics['recall']:.4f}")
    print(f"宏F1值:   {test_metrics['f1']:.4f}")
    print("=" * 50)

    swanlab.log({
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"]
    })

    swanlab.finish()

    with open("./label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    return test_metrics, label_map
