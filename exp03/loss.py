import torch
import torch.nn as nn


class SaliencyLoss(nn.Module):
    def __init__(self, weight_kld=1.0, weight_cc=0.5, weight_bce=1.0):
        """
        显著性预测的组合损失函数 (SOTA Standard)
        Loss = w1 * KLD_Loss + w2 * CC_Loss + w3 * BCE_Loss

        参数:
            weight_kld (float): KLD 损失的权重 (建议 1.0 - 10.0)
            weight_cc (float): CC 损失的权重 (建议 0.5 - 2.0)
            weight_bce (float): BCE 损失的权重 (建议 1.0)
        """
        super(SaliencyLoss, self).__init__()
        self.weight_kld = weight_kld
        self.weight_cc = weight_cc
        self.weight_bce = weight_bce
        self.bce = nn.BCELoss()  # 注意：输入必须已经经过 Sigmoid

    def forward(self, preds, targets):
        """
        preds:   模型输出 (B, 1, H, W)，取值范围必须是 [0, 1] (即经过 Sigmoid)
        targets: 真实标签 (B, 1, H, W)，取值范围 [0, 1]
        """
        # 1. BCE Loss (像素级约束)
        loss_bce = self.bce(preds, targets)

        # 2. CC Loss (相关性约束)
        # CC 越接近 1 越好，所以 Loss = 1 - CC
        loss_cc = 1.0 - self.cc_loss(preds, targets)

        # 3. KLD Loss (分布约束)
        loss_kld = self.kld_loss(preds, targets)

        # 总损失
        total_loss = (self.weight_bce * loss_bce +
                      self.weight_cc * loss_cc +
                      self.weight_kld * loss_kld)

        return total_loss, loss_bce, loss_cc, loss_kld

    def cc_loss(self, x, y):
        """
        计算皮尔逊相关系数 (Pearson Correlation Coefficient)
        """
        # 展平所有像素: (B, 1, H, W) -> (B, N)
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)

        # 减去均值
        x_mean = torch.mean(x, dim=1, keepdim=True)
        y_mean = torch.mean(y, dim=1, keepdim=True)
        x = x - x_mean
        y = y - y_mean

        # 计算分子 (协方差)
        numerator = torch.sum(x * y, dim=1)

        # 计算分母 (标准差乘积)
        denominator = torch.sqrt(torch.sum(x ** 2, dim=1) * torch.sum(y ** 2, dim=1)) + 1e-8

        # 计算 CC 并求 Batch 平均
        cc = torch.mean(numerator / denominator)
        return cc

    def kld_loss(self, pred, target):
        """
        计算 KL 散度 (Kullback-Leibler Divergence)
        注意：KLD 要求输入必须是“概率分布”，即一张图所有像素之和为 1
        """
        # 展平: (B, N)
        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)

        # 归一化为概率分布 (Sum = 1)
        # 加上 epsilon 防止除以 0
        pred_sum = torch.sum(pred, dim=1, keepdim=True) + 1e-8
        target_sum = torch.sum(target, dim=1, keepdim=True) + 1e-8

        pred_norm = pred / pred_sum
        target_norm = target / target_sum

        # KLD 公式: sum( P * log(P / Q) )
        # P是真值, Q是预测值
        eps = 1e-8
        kld = torch.sum(target_norm * torch.log((target_norm + eps) / (pred_norm + eps)), dim=1)

        return torch.mean(kld)
