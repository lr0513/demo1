# 基于BERT的中文新闻文本分类系统

## 一、项目简介
本项目基于 `bert-base-chinese` 预训练语言模型，实现今日头条新闻标题的15分类任务。采用 PyTorch 深度学习框架，遵循模块化、可复现的工程规范设计，支持训练过程可视化、超参数统一管理、验证集最优模型自动保存、早停机制等完整训练流程。

## 二、核心特性
- **模块化架构**：数据处理、模型定义、训练逻辑、评估工具完全解耦，便于维护和二次开发
- **配置化管理**：所有超参数、文件路径统一存放于JSON配置文件，调参无需修改业务代码
- **工程化训练**：集成随机种子固定、梯度裁剪、学习率预热衰减、早停机制等训练优化技巧
- **手写评价指标**：独立实现准确率、精确率、召回率、F1值等分类指标，不依赖第三方评估库
- **动态批次填充**：自定义Collator实现批次内动态padding，减少冗余计算，提升训练效率
- **可视化实验**：集成SwanLab实验可视化工具，自动记录训练曲线、超参数、系统资源，支持多实验对比
- **最优模型保存**：训练过程自动保存验证集表现最优的模型权重，避免过拟合影响最终结果

## 三、环境依赖
### 1. 环境要求
- Python >= 3.8
- PyTorch >= 1.10
- CUDA >= 11.0（推荐，CPU也可正常运行）

### 2. 一键安装依赖
```bash
pip install torch transformers swanlab numpy
```

## 四、项目目录结构
```text
project/
├── data/                    # 数据集目录
│   ├── train_3k.txt         # 训练集，共3000条
│   ├── dev_1k.txt           # 验证集，共1000条
│   └── test_1k.txt          # 测试集，共1064条
├── save_model/              # 最优模型权重保存目录（自动创建）
├── logs/                    # 训练日志目录（自动创建）
├── config.json              # 全局超参数配置文件
├── config.py                # 配置解析类，加载JSON参数
├── dataset.py               # 数据加载、自定义Dataset、批次处理
├── model.py                 # BERT分类模型定义
├── train.py                 # 训练主逻辑、验证、早停、模型保存
├── evaluate.py              # 模型评估推理逻辑
├── utils.py                 # 工具函数：种子固定、指标计算、文件工具
├── main.py                  # 项目统一入口
├── .gitignore               # Git忽略文件配置
└── README.md                # 项目说明文档
```

## 数据集说明
1. 数据集来源</br>今日头条中文新闻标题分类数据集，包含 15 个新闻类别的标题与关键词数据。
2. 数据格式</br>每行采用 _!_ 分隔符，共 5 个字段，格式如下：</br>
```text
新闻ID_!_编号_!_类别名称_!_新闻标题_!_关键词
```
3. 共 15 个新闻类别，具体如下：

| 类别名称 | 中文类别 |
| ---- | ---- |
| news_agriculture | 农业新闻 |
| news_car | 汽车新闻 |
| news_culture | 文化新闻 |
| news_edu | 教育新闻 |
| news_entertainment | 娱乐新闻 |
| news_finance | 财经新闻 |
| news_game | 游戏新闻 |
| news_house | 房产新闻 |
| news_military | 军事新闻 |
| news_sports | 体育新闻 |
| news_story | 故事新闻 |
| news_tech | 科技新闻 |
| news_travel | 旅游新闻 |
| news_world | 国际新闻 |
| stock | 股票资讯 |


## 五、实验结果
swanlab: 📁 View project at https://swanlab.cn/@lr0513/bert-toutiao-classify</br>
swanlab: 🚀 View run bert-base_maxlen128_dropout0.2 at 
https://swanlab.cn/@lr0513/bert-toutiao-classify/runs/1qu7ebkj</br>
**实验配置**：
- 预训练模型：bert-base-chinese
- 最大序列长度：128 
- 批次大小：16 
- 学习率：2e-5 
- Dropout：0.2 
- 早停耐心值：5

**核心指标**：
- 准确率: 0.8318
- 宏精确率: 0.8286
- 宏召回率: 0.8089
- 宏F1值:   0.8163

**结果分析**：

