import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from config import config
from model import CVAE

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp04.utils import plot_loss_curve

# --- 超参数设置 ---
BATCH_SIZE = config.BATCH_SIZE
LR = config.LEARNING_RATE
EPOCHS = config.EPOCHS
LATENT_DIM = config.LATENT_DIM
NUM_CLASSES = config.NUM_CLASSES
IMG_SIZE = config.IMG_SIZE
DEVICE = config.DEVICE
DATA_PATH = config.DATA_PATH
# --- 1. 数据准备 ---
# CIFAR-10 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    # VAE通常使用 Sigmoid 输出 [0,1]，所以这里不做均值标准差归一化到[-1,1]，
    # 而是保持 [0,1] 区间。如果使用 Tanh 激活，则需要 Normalize。
])

train_dataset = datasets.CIFAR10(root=DATA_PATH, train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root=DATA_PATH, train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 初始化模型
model = CVAE(LATENT_DIM, NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)


# --- 3. 损失函数 ---
# Loss = Reconstruction Loss + KL Divergence
def loss_function(recon_x, x, mu, logvar):
    # 重建损失：MSE (也可以尝试 BCE，但在彩色自然图像上MSE更常用)
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # KL 散度
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


# --- 4. 训练与主观评估 ---
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    return train_loss / len(train_loader.dataset)


def visualize_results(epoch):
    """
    主观评估：
    1. 保存固定噪声生成的图像，观察随 Epoch 的变化。
    2. 按类别生成图像（分析不同类别的生成效果）。
    """
    model.eval()
    with torch.no_grad():
        # A. 按类别生成 (每个类别生成 8 张)
        # 构造标签: 0,0,..,0, 1,1,..,1, ... 9,9,..,9
        n_per_class = 8
        labels = torch.cat([torch.tensor([i] * n_per_class) for i in range(NUM_CLASSES)])
        labels = labels.to(DEVICE)

        # 采样隐变量
        z = torch.randn(NUM_CLASSES * n_per_class, LATENT_DIM).to(DEVICE)

        sample = model.decode(z, labels).cpu()
        pic_path = os.path.join(config.RESULTS_PATH, f'sample_epoch_{epoch}.png')
        # 保存图片
        save_image(sample, pic_path, nrow=n_per_class)
        print(f"Saved generated images to results/sample_epoch_{epoch}.png")


def generate_images_for_eval(model, num_images=10000, output_dir='/root/autodl-tmp/generated_cifar10'):
    """生成大量图片用于计算 FID/IS"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 100
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            # 随机生成标签
            labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(DEVICE)
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            images = model.decode(z, labels).cpu()

            for j, img in enumerate(images):
                save_image(img, os.path.join(output_dir, f'{i + j}.png'))
    print(f"Generated {num_images} images to {output_dir}")


# --- 开始训练 ---
if __name__ == '__main__':
    print("Starting training...")
    loss_values = []
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(epoch)
        loss_values.append(avg_loss)
        # 每个 epoch 结束后进行主观评估采样
        visualize_results(epoch)
        if epoch % 10 == 0:
            save_path = os.path.join(config.CHECKPOINT_PATH, f'cvae_cifar10-{epoch}.pth')
            torch.save(model.state_dict(), save_path)
    plot_loss_curve(loss_values, "./training_loss_curve.png")
    print("Training finished.")
