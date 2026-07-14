# BERT‑base‑Chinese 新闻标题多分类项目
本项目基于Hugging‑Face提供的`bert-base-chinese`中文预训练模型，完成新闻标题的多分类任务。严格划分训练集、验证集、测试集，遵循深度学习标准实验流程；使用AdamW优化器更新模型参数，通过SwanLab工具记录训练损失与分类准确率，可视化模型收敛情况。程序自动判断设备环境，优先使用GPU加速训练，无GPU则使用CPU运行。

### 依赖库安装命令
```bash
pip install torch transformers scikit-learn swanlab
```

## 数据集
### 数据集格式
```text
新闻ID_!_数字编号_!_类别名称_!_新闻标题_!_关键词
```
### 文件说明
- train_3k.txt：训练数据集，用来更新模型权重
- dev_1k.txt：验证数据集，每一轮训练结束评估模型效果，观察过拟合情况
- test_1k.txt：测试数据集，训练全部结束后得到模型最终准确率

## 程序执行流程
1. 数据预处理阶段 
   - load_data()函数读取 txt 文件，提取标题与类别；
   - LabelEncoder将文本标签转为数字格式；
   - BertTokenizer把文本转换成模型可以识别的input_ids和attention_mask；
   - 自定义Dataset结合DataLoader实现分批加载样本数据。
2. 模型训练阶段
   - 加载 BERT 预训练模型，自动添加分类层用于多分类任务；
   - 开启model.train()训练模式，计算损失函数，通过反向传播更新模型参数；
3. 模型验证阶段
   - 开启model.eval()评估模式，借助torch.no_grad()关闭梯度计算节省显存；
   - 在验证集上计算损失和准确率，通过 SwanLab 保存各项实验指标；
4. 模型测试阶段
   - 全部训练轮次结束之后，在测试集上计算最终准确率，输出实验结果。

## SwanLab可视化查看
网址：https://swanlab.cn/@lr0513/demo1?utm_source=website_qr&utm_medium=qr_scan

按照lr→batch_size→epoch→max_length的顺序进行调参后可以得到：lr=2e-5,batch_size=32,epochs=4,max_length=32或128时，测试准确率最高，为83.18%