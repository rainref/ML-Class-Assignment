# config.py
import os
import sys

import torch


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout  # 记录原来的标准输出（屏幕）
        self.log = open(filename, "a", encoding='utf-8')  # 打开文件

    def write(self, message):
        self.terminal.write(message)  # 写到屏幕
        self.log.write(message)  # 写到文件
        self.log.flush()  # 立即刷新缓冲区，防止程序崩溃导致日志丢失

    def flush(self):
        # 必须实现这个方法，因为 python 内部会调用它
        self.terminal.flush()
        self.log.flush()


class config:
    # --- 1. 训练超参数 (Training Hyperparameters) ---
    BATCH_SIZE = 128  # 批大小
    EPOCHS = 50  # 训练轮数
    LEARNING_RATE = 1e-3  # 学习率
    SEED = 42  # 随机种子（保证结果可复现）

    # --- 2. 模型结构参数 (Model Architecture) ---
    LATENT_DIM = 256  # 隐变量维度 (Z vector size)
    NUM_CLASSES = 10  # 类别数 (CIFAR-10 为 10)
    IMG_SIZE = 32  # 图片尺寸
    CHANNELS = 3  # 图片通道数 (RGB)

    # --- 3. 路径配置 (Paths) ---
    DATA_PATH = '/root/autodl-tmp/CIFARdata'  # 数据集下载/存放路径
    RESULTS_PATH = '/root/autodl-tmp/results'  # 生成结果保存路径
    CHECKPOINT_PATH = '/root/autodl-tmp/checkpoints'  # 模型权重保存路径
    FINAL_MODEL_NAME = 'cvae_cifar10_{}.pth'  # 最终模型文件名

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 5. 自动创建必要的文件夹 ---
# 这样在 import config 时就会自动检查文件夹是否存在
os.makedirs(config.RESULTS_PATH, exist_ok=True)
os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)


# 打印配置信息 (可选)
def print_config():
    print("=" * 30)
    print("Current Configuration:")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Latent Dim: {config.LATENT_DIM}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("=" * 30)


print_config()
