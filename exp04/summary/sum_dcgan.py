import torch
from torchinfo import summary

from exp04.DCGAN.config import config  # 导入你的配置
from exp04.DCGAN.model import Generator, Discriminator  # 导入你的模型类

# --- 1. 准备虚拟数据 (Dummy Data) ---
batch_size = 64
z_dim = config.Z_DIM
num_classes = config.NUM_CLASSES
img_size = config.IMG_SIZE
channels = config.CHANNELS
device = config.DEVICE

# 构造数据
# A. 噪声 Z: (Batch, 100)
dummy_z = torch.randn(batch_size, z_dim).to(device)
# B. 标签 Labels: (Batch)
dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
# C. 图片 Images: (Batch, 3, 32, 32)
dummy_imgs = torch.randn(batch_size, channels, img_size, img_size).to(device)

# --- 2. 可视化生成器 (Generator) ---
print("\n" + "=" * 30 + " Generator Summary " + "=" * 30)
netG = Generator().to(device)

# Generator 的 forward(self, noise, labels)
summary(
    netG,
    input_data=[dummy_z, dummy_labels],  # 传入噪声和标签
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    depth=3,
    device=device
)

# --- 3. 可视化判别器 (Discriminator) ---
print("\n" + "=" * 30 + " Discriminator Summary " + "=" * 30)
netD = Discriminator().to(device)

# Discriminator 的 forward(self, img, labels)
summary(
    netD,
    input_data=[dummy_imgs, dummy_labels],  # 传入图片和标签
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    depth=3,
    device=device
)
