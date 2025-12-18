import matplotlib.pyplot as plt
import numpy as np


# 画损失曲线
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
