import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import swanlab

# 读取文本数据
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split('_!_')
            if len(parts) == 5:
                texts.append(parts[3])
                labels.append(parts[2])
    return texts, labels

train_texts, train_labels = load_data("data/train_3k.txt")
dev_texts, dev_labels = load_data("data/dev_1k.txt")
test_texts, test_labels = load_data("data/test_1k.txt")

# 标签转为数字
le = LabelEncoder()
train_y = le.fit_transform(train_labels)
dev_y = le.transform(dev_labels)
test_y = le.transform(test_labels)
num_labels = len(le.classes_)

# 构建BERT输入
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encode = tokenizer(text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": encode["input_ids"].flatten(),
            "attention_mask": encode["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

batch_size = 32
train_loader = DataLoader(NewsDataset(train_texts, train_y), batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(NewsDataset(dev_texts, dev_y), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(NewsDataset(test_texts, test_y), batch_size=batch_size, shuffle=False)

# 模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 4

# 可视化配置
swanlab.init(
    project="demo1",
    experiment_name="max_length=64",
    config={"lr":2e-5, "batch_size":32, "epochs":4}
)

# 训练过程
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    swanlab.log({"train_loss": train_loss})

    # 验证集评估
    model.eval()
    val_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            val_loss += outputs.loss.item()
            pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    val_loss /= len(dev_loader)
    val_acc = accuracy_score(trues, preds)
    swanlab.log({"val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch {epoch+1}, val_acc:{val_acc:.4f}")

# 测试集结果
model.eval()
test_pred, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attn_mask)
        pred = torch.argmax(outputs.logits, dim=1)
        test_pred.extend(pred.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_true, test_pred)
print(f"Test Accuracy:{test_acc:.4f}")