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
    # --- 训练超参数 ---
    BATCH_SIZE = 64  # GAN 建议用小一点的 Batch Size (64 或 128)
    EPOCHS = 200  # GAN 需要训练更久，CIFAR-10 建议 100-200 轮
    RESUME_EPOCH = -1  # 从xxx轮接续训练
    LR = 0.0002  # DCGAN 标准学习率
    BETA1 = 0.5  # Adam 的 beta1，DCGAN 论文建议 0.5 而不是默认的 0.9
    BETA2 = 0.999
    Z_DIM = 100  # 噪声维度
    NUM_CLASSES = 10
    IMG_SIZE = 32
    CHANNELS = 3

    # --- 路径 ---
    DATA_PATH = '/root/autodl-tmp/CIFARdata'
    RESULTS_PATH = '/root/autodl-tmp/results_gan'
    CHECKPOINT_PATH = '/root/autodl-tmp/checkpoints_gan'
    OUTPUT_DIR = '/root/autodl-tmp/generated_cifar10'  # 评估生成图像路径

    # --- 设备 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(config.RESULTS_PATH, exist_ok=True)
os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)


def print_config():
    print("=" * 30)
    print("Current Configuration:")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"BETA1: {config.BETA1}")
    print(f"Z Dim: {config.Z_DIM}")
    print(f"Learning Rate: {config.LR}")
    print("=" * 30)


print_config()
