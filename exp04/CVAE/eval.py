import os
import sys

import torch
from torchvision.utils import save_image

from config import config, Logger
from model import CVAE

# 获取当前脚本的绝对路径 (.../exp04/GAN)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (向上退两级: .../exp04/GAN -> .../exp04 -> .../ML-Class-Assignment)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# 将根目录加入 Python 搜索路径
sys.path.append(project_root)
from exp04.utils import fidelity_metric

# ==========================================
# 🚑【修复 PyTorch 2.6+ 兼容性问题的补丁】
# ==========================================
_original_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_torch_load


# ==========================================

def generate_cvae_images(model, gen_path, total_images=10000, batch_size=100):
    """
    CVAE 专用生成函数
    """
    os.makedirs(gen_path, exist_ok=True)
    model.eval()

    print(f"🚀 CVAE 开始生成 {total_images} 张图像到: {gen_path} ...")
    count = 0

    with torch.no_grad():
        while count < total_images:
            current_batch = min(batch_size, total_images - count)

            # 1. 采样隐变量 z
            z = torch.randn(current_batch, config.LATENT_DIM).to(config.DEVICE)

            # 2. 随机生成标签 (0-9) 以保证类别均衡
            labels = torch.randint(0, config.NUM_CLASSES, (current_batch,)).to(config.DEVICE)

            # 3. 解码 (使用 decode 方法，跳过 encoder)
            # 注意：CVAE 输出通常已经是 Sigmoid [0, 1]
            gen_imgs = model.decode(z, labels)

            # 4. 保存
            for i in range(current_batch):
                save_image(gen_imgs[i], os.path.join(gen_path, f"{count}.png"))
                count += 1

            sys.stdout.write(f"\r进度: [{count}/{total_images}]")
            sys.stdout.flush()

    print("\n✅ 生成完毕！")


def evaluate_cvae(epoch, total_images=10000):
    print(f"🔍 准备评估 CVAE 模型 (Epoch {epoch})...")

    # 1. 权重路径 (假设 CVAE 的权重保存在 checkpoints 根目录或专门文件夹)
    # 请根据你的实际保存路径修改这里，例如 config.CHECKPOINT_PATH
    # 假设文件名是 cvae_epoch_{epoch}.pth
    checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f"cvae_cifar10-{epoch}.pth")

    if not os.path.exists(checkpoint_path):
        # 尝试找找有没有 final 模型
        checkpoint_path = os.path.join(config.CHECKPOINT_PATH, "cvae_cifar10_final.pth")
        if not os.path.exists(checkpoint_path):
            print(f"❌ 错误：找不到权重文件: {checkpoint_path}")
            return

    # 2. 加载模型
    print(f"📥 正在加载权重: {checkpoint_path}")
    model = CVAE(latent_dim=config.LATENT_DIM, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))

    # 3. 生成图片
    gen_folder = f"eval_cvae_epoch_{epoch}"
    gen_path = os.path.join(config.RESULTS_PATH, gen_folder)
    generate_cvae_images(model, gen_path, total_images=total_images)

    # 4. 计算指标
    print("⏳ 正在调用 torch-fidelity 计算指标 (IS/FID/KID)...")
    try:
        metrics_dict = fidelity_metric(gen_path, config.DATA_PATH)

        print("\n" + "=" * 50)
        print(f"📊 CVAE 评估结果 (Epoch {epoch})")
        print("=" * 50)
        print(
            f"Inception Score: {metrics_dict['inception_score_mean']:.4f} ± {metrics_dict['inception_score_std']:.4f}")
        print(f"FID:             {metrics_dict['frechet_inception_distance']:.4f}")
        print(f"KID:             {metrics_dict['kernel_inception_distance_mean']:.4f}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"❌ 计算出错: {e}")


if __name__ == "__main__":
    sys.stdout = Logger("training_log.txt")
    evaluate_cvae(epoch=50, total_images=10000)  # 你可以修改 epoch 参数
