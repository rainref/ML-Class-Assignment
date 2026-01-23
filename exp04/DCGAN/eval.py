import os
import sys

import torch

from config import config
from model import Generator

# 获取当前脚本的绝对路径 (.../exp04/GAN)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (向上退两级: .../exp04/GAN -> .../exp04 -> .../ML-Class-Assignment)
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
# 将根目录加入 Python 搜索路径
sys.path.append(project_root)
from exp04.utils import fidelity_metric, generate_images_to_folder

# ==========================================
# 🚑【修复 PyTorch 2.6+ 兼容性问题的补丁】
# 必须加在 eval.py 里，因为 torch-fidelity 也会在这里被调用
# ==========================================
_original_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_torch_load


# ==========================================


def evaluate_model(epoch, total_images=10000):
    """
    加载指定 Epoch 的模型并计算 IS, FID, KID
    """
    print(f"🔍 准备评估 Epoch {epoch} 的模型...")

    # 1. 定义路径
    checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f"generator_epoch_{epoch}.pth")
    # 生成图片的临时文件夹
    gen_folder_name = f"eval_temp_epoch_{epoch}"
    # 这里建议把生成的图放在 tmp 目录，或者 results 目录
    gen_path = os.path.join(config.RESULTS_PATH, gen_folder_name)

    # 2. 检查权重文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误：找不到权重文件: {checkpoint_path}")
        return

    # 3. 加载模型
    print(f"📥 正在加载模型权重: {checkpoint_path}")
    netG = Generator().to(config.DEVICE)

    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
    netG.load_state_dict(state_dict)

    # 切换到评估模式
    netG.eval()

    # 4. 生成图片 (用于 FID 计算)
    # 如果文件夹已经存在且图片数量够，可以选择跳过生成(节省时间)，这里默认重新生成以防万一
    print(f"🎨 正在生成 {total_images} 张图片到 {gen_path} ...")
    generate_images_to_folder(netG, gen_path, config.Z_DIM, config.NUM_CLASSES, config.DEVICE)

    # 5. 计算指标
    print("jj 正在调用 torch-fidelity 计算指标 (可能需要几分钟)...")
    try:
        metrics_dict = fidelity_metric(gen_path, config.DATA_PATH)

        # 6. 打印结果
        print("\n" + "=" * 40)
        print(f"DCGAN 📊 评估结果 (Epoch {epoch})")
        print("=" * 40)
        print(
            f"Inception Score: {metrics_dict['inception_score_mean']:.4f} ± {metrics_dict['inception_score_std']:.4f}")
        print(f"FID:             {metrics_dict['frechet_inception_distance']:.4f}")
        print(f"KID:             {metrics_dict['kernel_inception_distance_mean']:.4f}")
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"❌ 计算指标时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    evaluate_model(epoch=200, total_images=10000)
