import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_fidelity
from torchvision.utils import save_image


def sample_images(epoch, generator, num_classes, device, results_path, z_dim):
    """按类别保存生成的图片"""
    generator.eval()
    with torch.no_grad():
        # 每个类别生成 8 张
        n_row = 10
        # 构造标签: 0~9 每个重复 8 次
        labels = torch.LongTensor(np.array([num for num in range(num_classes) for _ in range(n_row)])).to(
            device)
        z = torch.randn(num_classes * n_row, z_dim).to(device)

        gen_imgs = generator(z, labels)

        # 因为生成范围是 [-1, 1]，保存前需要反归一化到 [0, 1]
        # (img + 1) / 2
        save_image(gen_imgs.data, os.path.join(results_path, f"epoch_{epoch}.png"), nrow=n_row, normalize=True)
    generator.train()


def plot_loss_curve(loss_values, pic_path):
    """
    简化版损失曲线绘制函数

    Parameters:
    -----------
    loss_values : List[float]
        损失值列表
    """
    # 创建图表
    plt.figure(figsize=(10, 5))

    # 生成步数列表
    steps = list(range(1, len(loss_values) + 1))

    # 绘制损失曲线
    plt.plot(steps, loss_values, 'b-', linewidth=1.5, alpha=0.8)
    plt.title('Training Loss Curve', fontsize=14, pad=20)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # 计算统计信息
    final_loss = loss_values[-1]
    min_loss = min(loss_values)
    max_loss = max(loss_values)
    mean_loss = np.mean(loss_values)

    # 在图表上添加统计信息
    stats_text = (f"Final Loss: {final_loss:.4f}\n"
                  f"Min Loss: {min_loss:.4f}\n"
                  f"Mean Loss: {mean_loss:.4f}")

    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 verticalalignment='top', fontsize=10)

    plt.tight_layout()

    plt.savefig(pic_path)
    plt.show()

    # 打印统计信息
    print(f"总训练步数: {len(loss_values)}")
    print(f"最终Loss: {final_loss:.4f}")
    print(f"最小Loss: {min_loss:.4f}")
    print(f"平均Loss: {mean_loss:.4f}")


def fidelity_metric(genereated_images_path, data_path):
    """
    使用fidelity package计算所有的生成相关的指标，输入生成图像路径和真实图像路径
    isc: inception score
    kid: kernel inception distance
    fid: frechet inception distance
    """
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=genereated_images_path,
        input2='cifar10-val',
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        verbose=False,
        datasets_root=data_path
    )
    print(f"Inception Score: {metrics_dict['inception_score_mean']:.4f}")
    print(f"FID: {metrics_dict['frechet_inception_distance']:.4f}")
    print(f"KID: {metrics_dict['kernel_inception_distance_mean']:.4f}")
    return metrics_dict


def generate_images_to_folder(generator, gen_path, z_dim, num_classes, device, total_images=10000):
    """
    使用生成器生成指定数量的图片并保存到文件夹，用于FID计算。

    Args:
        generator: 训练好的生成器模型
        gen_path: 图片保存的文件夹路径
        total_images: 需要生成的总数量 (默认 10000，对应 CIFAR-10 测试集大小)
    """
    # 1. 创建目录
    os.makedirs(gen_path, exist_ok=True)

    # 2. 切换到评估模式 (对于 BatchNorm/Dropout 很重要)
    generator.eval()

    print(f"🚀 开始生成 {total_images} 张图像到: {gen_path} ...")

    count = 0
    # 生成时的 Batch Size 可以设置大一点以提高速度，只要显存够用
    gen_batch_size = 100

    with torch.no_grad():
        while count < total_images:
            # 计算当前批次需要生成的数量（防止最后一次超出 total_images）
            current_batch_size = min(gen_batch_size, total_images - count)

            # A. 构造输入
            z = torch.randn(current_batch_size, z_dim).to(device)
            # 随机生成标签 (0-9)
            labels = torch.randint(0, num_classes, (current_batch_size,)).to(device)

            # B. 生成图像
            gen_imgs = generator(z, labels)

            # C. 【关键】反归一化 Denormalization
            # Generator 输出是 Tanh [-1, 1]，我们需要 [0, 1] 才能保存为 PNG
            gen_imgs = (gen_imgs + 1) / 2.0

            # 钳位以防数值溢出 (可选，但推荐)
            gen_imgs.clamp_(0, 1)

            # D. 循环保存当前批次的每一张图
            for i in range(current_batch_size):
                file_name = f"{count}.png"
                save_path = os.path.join(gen_path, file_name)
                save_image(gen_imgs[i], save_path)
                count += 1

            # 打印进度条
            sys.stdout.write(f"\r进度: [{count}/{total_images}] ({(count / total_images) * 100:.1f}%)")
            sys.stdout.flush()

    print("\n✅ 生成完毕！")


def plot_evaluation_dashboard(loss_values, fid_values, is_mean_values, is_std_values=None, save_path=None):
    """
    绘制模型评估面板：Loss, Inception Score, FID 以及 Loss vs FID 相关性。

    参数:
    -----
    loss_values : list or np.array
        每个 epoch 的 Training Loss (例如 MSE 或 G_Loss)
    fid_values : list or np.array
        每个 epoch 的 FID 分数
    is_mean_values : list or np.array
        每个 epoch 的 Inception Score 均值
    is_std_values : list or np.array, optional
        每个 epoch 的 Inception Score 标准差 (默认为 0)
    save_path : str, optional
        图片保存路径 (例如 'results/metrics.png')。如果不传，则直接显示。
    """

    # 1. 数据准备
    # --- 关键修正 1: 生成两套 X 轴坐标 ---
    # Set A: 用于 Loss (总长度，例如 100)
    epochs_loss = np.arange(1, len(loss_values) + 1)

    # Set B: 用于 FID/IS (稀疏长度，例如 10)
    # 计算步长：例如 100 // 10 = 10
    if len(fid_values) > 0:
        step = len(loss_values) // len(fid_values)
        # 生成 [10, 20, ..., 100]
        epochs_eval = np.arange(step, len(loss_values) + 1, step)
        # 【双重保险】截断多余的坐标，确保 x 和 y 长度绝对一致
        epochs_eval = epochs_eval[:len(fid_values)]
    else:
        epochs_eval = np.array([])

    if is_std_values is None:
        is_std_values = np.zeros_like(is_mean_values)

    # 确保输入是 numpy 数组以便绘图
    loss_values = np.array(loss_values)
    fid_values = np.array(fid_values)
    is_mean_values = np.array(is_mean_values)

    # 2. 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    # --- 图 1: Training Loss (左上) ---
    ax1 = axes[0, 0]
    ax1.plot(epochs_loss, loss_values, label='Training Loss', color='#377eb8', linewidth=1.5)
    ax1.set_title('Training Loss Over Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- 图 2: Inception Score (右上) ---
    ax2 = axes[0, 1]
    # --- 关键修正 2: 使用 epochs_eval (短) ---
    if len(epochs_eval) == len(is_mean_values):
        ax2.errorbar(epochs_eval, is_mean_values, yerr=is_std_values, fmt='-o',
                     label='Inception Score', color='#377eb8', ecolor='orange',
                     markersize=4, elinewidth=1, capsize=2)
    else:
        print(
            f"⚠️ 警告: IS 数据长度 ({len(is_mean_values)}) 与计算出的 Epoch 长度 ({len(epochs_eval)}) 不匹配，跳过绘图。")

    ax2.set_title('Inception Score Over Epochs (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Inception Score')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # --- 图 3: Frechet Inception Distance (左下) ---
    ax3 = axes[1, 0]
    # --- 关键修正 3: 使用 epochs_eval (短) ---
    if len(epochs_eval) == len(fid_values):
        ax3.plot(epochs_eval, fid_values, label='FID', color='#e41a1c', linewidth=1.5, marker='.')

    ax3.set_title('Frechet Inception Distance Over Epochs (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('FID')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()

    # --- 图 4: Loss vs FID Correlation (右下) ---
    ax4 = axes[1, 1]
    # 【核心逻辑】处理长度不一致
    if len(loss_values) != len(fid_values):
        # 计算步长，例如 100 // 10 = 10
        align_step = len(loss_values) // len(fid_values)

        # 对 Loss 进行切片：从第 step-1 个开始，每隔 step 取一个
        # [:len(fid_values)] 是为了防止除不尽导致的长度溢出
        aligned_loss = loss_values[align_step - 1:: align_step][:len(fid_values)]
    else:
        aligned_loss = loss_values

    # 再次检查长度，防止 crash
    if len(aligned_loss) == len(fid_values):
        ax4.scatter(aligned_loss, fid_values, label='Loss vs FID', color='#3d8026', alpha=0.8, s=30)

    ax4.set_title('Loss vs FID Correlation', fontsize=12, fontweight='bold')
    ax4.set_xlabel('MSE Loss (Sampled)')
    ax4.set_ylabel('FID')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend()

    # 3. 布局调整与保存
    plt.tight_layout()

    if save_path:
        # 自动创建目录（如果不存在）
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 评估图表已保存至: {save_path}")
    else:
        plt.show()

    plt.close()  # 关闭画布释放内存
