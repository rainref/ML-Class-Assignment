import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import config, Logger
from model import Generator, Discriminator, weights_init_normal

# ==========================================
# 🚑【修复 PyTorch 2.6+ 兼容性问题的补丁】开始
# ==========================================
# 强制让 torch.load 默认使用 weights_only=False，恢复旧版本行为
_original_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    # 如果调用方没有指定 weights_only，则手动设置为 False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_torch_load
# ==========================================
# 🚑【补丁】结束
# ==========================================
# 获取当前脚本的绝对路径 (.../exp04/GAN)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (向上退两级: .../exp04/GAN -> .../exp04 -> .../ML-Class-Assignment)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# 将根目录加入 Python 搜索路径
sys.path.append(project_root)
from exp04.utils import sample_images, plot_loss_curve, fidelity_metric, generate_images_to_folder, \
    plot_evaluation_dashboard

# --- 1. 数据加载 (重点修改：归一化到 -1 ~ 1) ---
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 重要！
])

train_dataset = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# --- 2. 初始化模型 ---
generator = Generator().to(config.DEVICE)
discriminator = Discriminator().to(config.DEVICE)

# 应用权重初始化
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# --- 3. 优化器与损失函数 ---
adversarial_loss = nn.BCELoss()  # 二分类交叉熵

optimizer_G = optim.Adam(generator.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))


# --- 4. 训练循环 ---
def train():
    print("开始 GAN 训练...")
    start_epoch = 1
    if config.RESUME_EPOCH > 0:
        print(f"🔄 正在加载第 {config.RESUME_EPOCH} 轮的权重以接续训练...")

        # 构造路径
        g_path = os.path.join(config.CHECKPOINT_PATH, f"generator_epoch_{config.RESUME_EPOCH}.pth")
        d_path = os.path.join(config.CHECKPOINT_PATH, f"discriminator_epoch_{config.RESUME_EPOCH}.pth")
        if os.path.exists(g_path) and os.path.exists(d_path):
            # 加载参数 (使用 map_location 防止设备不匹配)
            generator.load_state_dict(torch.load(g_path, map_location=config.DEVICE))
            discriminator.load_state_dict(torch.load(d_path, map_location=config.DEVICE))
            print("✅ 模型权重加载成功！")
            start_epoch = config.RESUME_EPOCH + 1
        else:
            print(f"❌ 错误：在 {config.CHECKPOINT_PATH} 下未找到第 {config.RESUME_EPOCH} 轮的权重文件！")
            print("将从头开始训练...")
            # 如果加载失败，应用初始化
            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)
    else:
        print("✨ 从头开始训练 (随机初始化)...")
        # 只有在从头训练时才应用随机初始化，否则会覆盖掉加载的权重
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    print(f"开始 GAN 训练 (从 Epoch {start_epoch} 到 {config.EPOCHS})...")

    g_loss_value = []
    d_loss_value = []
    fid_list = []
    kid_list = []
    is_mean_list = []
    is_std_list = []
    for epoch in range(start_epoch, config.EPOCHS + 1):
        for i, (imgs, labels) in enumerate(train_loader):
            # if i == 0:
            #     print(f"Min: {imgs.min().item()}, Max: {imgs.max().item()}")

            # 配置输入
            batch_size = imgs.shape[0]
            real_imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # 定义标签 (1: Real, 0: Fake)
            valid = torch.ones(batch_size, 1).to(config.DEVICE)
            fake = torch.zeros(batch_size, 1).to(config.DEVICE)
            valid_smooth = torch.full((batch_size, 1), 0.9).to(config.DEVICE)

            # -----------------
            #  训练 Generator
            # -----------------
            optimizer_G.zero_grad()

            # 采样噪声和随机类别
            z = torch.randn(batch_size, config.Z_DIM).to(config.DEVICE)
            gen_labels = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(config.DEVICE)

            # 生成图片
            gen_imgs = generator(z, gen_labels)

            # Loss: 希望 Discriminator 认为这些生成的图片是 Valid (1)
            # D(G(z)) -> 1
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  训练 Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # 1. 真实图片 Loss
            real_pred = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid_smooth)

            # 2. 生成图片 Loss (使用 .detach() 防止梯度传回 G)
            fake_pred = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # 总 D Loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{config.EPOCHS}] [Batch {i}/{len(train_loader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
            is_last_batch = (i == len(train_loader) - 1)
            if is_last_batch:
                g_loss_value.append(g_loss.item())
                d_loss_value.append(d_loss.item())
        # --- 每个 Epoch 结束后的可视化 ---
        sample_images(epoch, generator, config.NUM_CLASSES, config.DEVICE, config.RESULTS_PATH, config.Z_DIM)
        print(f"正在评估第 {epoch} 轮...")
        # A. 先生成图片保存到文件夹
        gen_path = os.path.join(config.OUTPUT_DIR, f"eval_epoch_{epoch}")
        generate_images_to_folder(generator, gen_path, config.Z_DIM, config.NUM_CLASSES, config.DEVICE)
        generator.train()

        # B. 调用你的评估函数
        metrics = fidelity_metric(gen_path, config.DATA_PATH)

        # C. 【核心】提取参数并存入列表
        fid_list.append(metrics['frechet_inception_distance'])
        kid_list.append(metrics['kernel_inception_distance_mean'])
        is_mean_list.append(metrics['inception_score_mean'])
        is_std_list.append(metrics['inception_score_std'])  # 这里拿到了 std

        # 保存模型
        if epoch % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(config.CHECKPOINT_PATH, f"generator_epoch_{epoch}.pth"))
            torch.save(discriminator.state_dict(),
                       os.path.join(config.CHECKPOINT_PATH, f"discriminator_epoch_{epoch}.pth"))

    print("训练结束，正在绘图...")
    # 将其存入日志文件，打印这些值
    print('g_loss_value:', g_loss_value)
    print('d_loss_value:', d_loss_value)
    print('fid_list:', fid_list)
    print('kid_list:', kid_list)
    print('is_mean_list:', is_mean_list)
    print('is_std_list:', is_std_list)
    plot_evaluation_dashboard(
        loss_values=g_loss_value,
        fid_values=fid_list,
        is_mean_values=is_mean_list,  # 对应参数 1
        is_std_values=is_std_list,  # 对应参数 2
        save_path='./final_evaluation.png'
    )
    plot_loss_curve(g_loss_value, './g_loss_curve.png')
    plot_loss_curve(d_loss_value, './d_loss_curve.png')


if __name__ == "__main__":
    sys.stdout = Logger("training_log.txt")
    train()
