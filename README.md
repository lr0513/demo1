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
├── config.json              # 实验超参数配置文件，可存放多份json做对比实验
├── config.py                # 配置解析类，加载JSON参数，不再导出全局cfg
├── dataset.py               # 数据加载、自定义Dataset、批次collate处理
├── model.py                 # BERT分类模型定义，超参由构造函数传入
├── train.py                 # 训练主逻辑、验证、早停、最优模型保存
├── evaluate.py              # 模型评估函数
├── utils.py                 # 工具函数：种子固定、手写分类指标计算、文件夹创建
├── predict.py               # 单条样本预测逻辑，自动读取label_map.json
├── main.py                  # 项目统一入口，argparse解析配置路径，训练+交互式预测
├── label_map.json           # 训练后自动生成，类别-数字id映射，预测阶段使用
├── .gitignore               # Git忽略文件配置，忽略.idea、模型权重、日志等
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

## 运行方式
本项目使用argparse命令行参数指定配置文件，不可直接点击运行main.py，必须传入--config_path参数
### 方式一：终端运行
```bash
# 使用根目录下config.json配置
python main.py --config_path config.json

# 切换其他实验配置文件
# python main.py --config_path configs/exp_bert-base-chinese_lr2e-5_bs16_len128_0704.json
```
### 方式二：PyCharm运行配置
1. Edit Configurations → 选中main.py 
2. 在Parameters填写：--config_path config.json 
3. 保存后点击运行；训练结束自动进入交互式预测，输入新闻标题得到分类，输入exit退出程序。
## 六、实验结果
**实验配置**：
- 预训练模型：bert-base-chinese
- 最大序列长度：128 
- 批次大小：16 
- 学习率：2e-5 
- Dropout：0.2 
- 早停耐心值：5
> 测试集评估：训练完成加载验证集最优权重进行测试集评估，不使用最有一轮epoch模型，避免高估模型效果

### 训练结果分析（对应SwanLab可视化图表）
swanlab: 📁 View project at https://swanlab.cn/@lr0513/bert-toutiao-classify</br>
swanlab: 🚀 View run bert-base_maxlen128_dropout0.2 at 
https://swanlab.cn/@lr0513/bert-toutiao-classify/runs/gbc7vnzm</br>
</br>从训练可视化曲线可以观察到：</br>
1. **训练损失**：训练损失快速下降，在第10个epoch之后趋于平稳，模型充分拟合训练数据。
2. **验证集指标（准确率、精确率、召回率、F1）**：前期快速上升，后期在小范围内震荡，没有出现明显持续下降，无严重过拟合现象；指标存在小幅波动属于小数据集训练的正常现象。
3. **学习率曲线**：执行预热后线性衰减策略，前期学习率小幅上升预热，之后逐步降低，符合 BERT 微调的标准学习率调度方案，有助于模型稳定收敛。
> 综合曲线表现：模型收敛状态健康，因此保存验证集最优权重用于测试集评估，可以得到可靠的泛化结果。

**核心指标**：
- 准确率: 0.8318
- 宏精确率: 0.8286
- 宏召回率: 0.8089
- 宏F1值: 0.8163
> 结果分析：验证集最优指标与测试集指标差距很小，模型泛化性能良好，无严重过拟合。少量类别存在识别混淆，主要来自类别边界模糊、样本不均衡问题。

![](https://obsidian-1322827540.cos.ap-guangzhou.myqcloud.com/img/cb39d6024d5895faa2990bc49f9ab1e7.png)
