import matplotlib.pyplot as plt
from train import plot_loss_curve
# 提取的Loss值
loss_values = [
    0.9661, 0.2929, 0.2055, 0.1518, 0.1474, 0.1445, 0.1411, 0.1191, 0.0972,
    0.0666, 0.0759, 0.0779, 0.0678, 0.0764, 0.0732, 0.0680, 0.0687, 0.0767,
    0.0505, 0.0487, 0.0473, 0.0977, 0.0870, 0.2135, 0.1757, 0.2609, 0.1702,
    0.0896, 0.0575, 0.0643, 0.0591, 0.0660, 0.0658, 0.0640, 0.0504, 0.0737,
    0.0422, 0.0367, 0.0821, 0.0568, 0.0477, 0.0422, 0.0422, 0.0363, 0.0551
]

# plot_loss_curve(loss_values)


# 创建步数列表
steps = list(range(1, len(loss_values) + 1))

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(steps, loss_values, 'b-', linewidth=1, alpha=0.7)
plt.title('Training Loss Over Steps', fontsize=14)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# 标记epoch边界
epoch_boundaries = [9, 18, 27, 36]  # 每个epoch结束的位置
for boundary in epoch_boundaries:
    plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

plt.tight_layout()
#plt.show()
plt.savefig('./training_loss_curve.png')
# 打印一些统计信息
print(f"总训练步数: {len(loss_values)}")
print(f"最终Loss值: {loss_values[-1]:.4f}")
print(f"最小Loss值: {min(loss_values):.4f}")
print(f"最大Loss值: {max(loss_values):.4f}")
print(f"平均Loss值: {sum(loss_values)/len(loss_values):.4f}")