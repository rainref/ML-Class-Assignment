# 实验一

[![ResNet Paper](https://img.shields.io/badge/arXiv-ResNet-b31b1b.svg)](https://arxiv.org/abs/1512.03385)

## 实验内容

使用深度残差网络ResNet解决MNIST手写数字分类问题。

## 项目结构

```
ML-Class-Assignment/
├── exp01/                   # 实验一：手写数字识别
│   ├── results/             # 结果保存
│   ├── model.py             # 模型定义文件
│   ├── train.py             # 模型训练脚本
│   ├── eval.py              # 模型评估脚本
|   ├── step_loss.py         # step loss绘图
│   |__ model_visualization.py     # 模型结构可视化
│__ README.md           # 项目说明文档
```

## 实验环境

```
# 超参数设置
batch_size = 64
learning_rate = 0.001
epochs = 5
optimizer = AdamW
loss_function = CrossEntropyLoss
# 数据预处理  
transform = transforms.Compose([  
    transforms.Resize((28, 28)),  # 确保尺寸一致  
    transforms.ToTensor(),  # 转换为Tensor  
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]  
])
```

## 实验结果

- 训练集准确率：99.09%
- 测试集准确率：98.73%
- 损失函数曲线：

<div align="center">
  <img src="results/training_loss_curve.png" width="45%" />
  <img src="results/training_batch_loss_curve.png" width="45%" />
</div>

- 分类混淆矩阵

<div align="center">
  <img src="results/confusion_matrix.png" width="50%" alt="分类混淆矩阵"/>
</div>
