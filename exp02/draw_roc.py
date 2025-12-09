import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix,
                             classification_report, accuracy_score)


def draw_roc_pr_curves(y_test, y_pred_prob, y_pred):
    """
    绘制ROC曲线和PR曲线，并显示相关评估指标
    :param y_test: 真实标签
    :param y_pred_prob: 预测为正类的概率
    :param y_pred: 预测的类别标签
    """
    # 1. 确保输入为numpy数组
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)[:, 1]  # 获取正类的预测概率
    # 3. 计算ROC曲线和AUC
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # 4. 计算PR曲线和平均精确率
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
    average_precision = average_precision_score(y_test, y_pred_prob)

    # 5. 创建可视化图表
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # 5.1 ROC曲线
    ax1 = axes[0, 0]
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 5.2 PR曲线
    ax2 = axes[0, 1]
    ax2.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {average_precision:.3f})')
    # 随机模型的PR曲线（正类比例作为基线）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    ax2.plot([0, 1], [baseline, baseline], color='red', lw=2,
             linestyle='--', label=f'Random (AP = {baseline:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # 5.3 阈值与指标关系图
    ax3 = axes[0, 2]
    ax3.plot(thresholds_roc[1:], tpr[1:], 'b-', label='True Positive Rate', lw=2)
    ax3.plot(thresholds_roc[1:], 1 - fpr[1:], 'r-', label='True Negative Rate', lw=2)
    ax3.set_xlabel('Decision Threshold')
    ax3.set_ylabel('Rate')
    ax3.set_title('Metrics vs. Threshold')
    ax3.legend(loc="lower left")
    ax3.grid(True, alpha=0.3)

    # 5.4 混淆矩阵热图
    ax4 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    im = ax4.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax4.set_title(f'Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
    plt.colorbar(im, ax=ax4)
    tick_marks = np.arange(2)
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels(['Pred 0', 'Pred 1'])
    ax4.set_yticklabels(['True 0', 'True 1'])
    # 在热图中添加数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    # 5.5 概率分布直方图
    ax5 = axes[1, 1]
    ax5.hist(y_pred_prob[y_test == 0], bins=30, alpha=0.7,
             label='Class 0', color='blue', density=True)
    ax5.hist(y_pred_prob[y_test == 1], bins=30, alpha=0.7,
             label='Class 1', color='orange', density=True)
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Probability Distribution by True Class')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 5.6 评估指标总结
    # ax6 = axes[1, 2]
    # ax6.axis('off')
    # # 生成分类报告
    # report = classification_report(y_test, y_pred, output_dict=True)
    # summary_text = f"""Model Performance Summary
    #
    # Overall Metrics:
    # - Accuracy: {report['accuracy']:.3f}
    # - ROC AUC: {roc_auc:.3f}
    # - Avg Precision: {average_precision:.3f}
    #
    # Class 0:
    # - Precision: {report['0']['precision']:.3f}
    # - Recall: {report['0']['recall']:.3f}
    # - F1-score: {report['0']['f1-score']:.3f}
    #
    # Class 1:
    # - Precision: {report['1']['precision']:.3f}
    # - Recall: {report['1']['recall']:.3f}
    # - F1-score: {report['1']['f1-score']:.3f}
    #
    # 最佳阈值信息:
    # - Youden指数最佳点: {thresholds_roc[np.argmax(tpr - fpr)]:.3f}
    # - 最大F1分数: {2 * (precision * recall) / (precision + recall + 1e-10)}
    # """
    # ax6.text(0.05, 0.95, summary_text, fontsize=10,
    #          verticalalignment='top', fontfamily='monospace')
    fig.delaxes(axes[1, 2])
    plt.tight_layout()
    plt.savefig('./roc-pr-worse.png')
    plt.show()

    # 6. 单独显示最佳阈值点
    # 使用Youden指数找到最佳阈值
    youden_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds_roc[youden_idx]
    best_fpr = fpr[youden_idx]
    best_tpr = tpr[youden_idx]

    print(f"\n{'=' * 50}")
    print("ROC曲线最佳阈值分析:")
    print(f"{'=' * 50}")
    print(f"最佳阈值 (Youden指数): {best_threshold:.4f}")
    print(f"对应的FPR: {best_fpr:.4f}")
    print(f"对应的TPR: {best_tpr:.4f}")
    print(f"对应的特异度: {1 - best_fpr:.4f}")

    # 7. 使用最佳阈值重新预测
    y_pred_optimized = (y_pred_prob >= best_threshold).astype(int)
    print(f"\n使用最佳阈值 {best_threshold:.4f} 后的性能:")
    print(classification_report(y_test, y_pred_optimized))


# 画损失曲线
def plot_loss_curve(loss_values):
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

    plt.savefig('./training_loss_curve_1.png')
    plt.show()

    # 打印统计信息
    print(f"总训练步数: {len(loss_values)}")
    print(f"最终Loss: {final_loss:.4f}")
    print(f"最小Loss: {min_loss:.4f}")
    print(f"平均Loss: {mean_loss:.4f}")
