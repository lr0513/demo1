import torch
from utils import calculate_metrics


def evaluate(model, data_loader, device, num_classes: int):
    """
    在指定数据集上评估模型效果（验证集/测试集通用）
    """
    model.eval()

    all_true = []  # 存放所有样本【真实标签】
    all_pred = []  # 存放所有样本【模型预测标签】
    total_loss = 0.0  # 累加整个数据集所有batch的loss

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += outputs["loss"].item()
            preds = torch.argmax(outputs["logits"], dim=1)

            all_pred.extend(preds.cpu().numpy().tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    if len(data_loader) > 0:
        avg_loss = total_loss / len(data_loader)
    else:
        avg_loss = 0.0

    # 传入实际类别数计算指标
    metrics = calculate_metrics(all_true, all_pred, num_classes=num_classes)
    metrics["loss"] = avg_loss

    return metrics
