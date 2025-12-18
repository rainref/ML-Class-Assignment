import numpy as np


def normalize_map(s_map):
    """
    Min-Max 归一化: 将图像数据映射到 [0, 1] 区间
    """
    s_map = s_map.astype(np.float32)
    min_val = np.min(s_map)
    max_val = np.max(s_map)

    eps = np.finfo(float).eps
    if max_val - min_val == 0:
        return s_map

    return (s_map - min_val) / (max_val - min_val + eps)


def normalize_density(s_map):
    """
    概率密度归一化: 使图像所有像素之和为 1
    (专门用于 KLD 计算)
    """
    s_map = s_map.astype(np.float32)
    s_sum = np.sum(s_map)

    eps = np.finfo(float).eps
    if s_sum == 0:
        return s_map

    return s_map / (s_sum + eps)


def cal_cc(gt, pred):
    """
    计算皮尔逊相关系数 (Pearson Correlation Coefficient, CC)
    衡量线性相关性，值范围 [-1, 1]，越接近 1 越好。

    参数:
        gt: Ground Truth 显著图 (H, W) 或 (N,)
        pred: 预测显著图 (H, W) 或 (N,)
    """
    # 确保展平
    gt = gt.flatten().astype(np.float32)
    pred = pred.flatten().astype(np.float32)

    # 简单的标准化 (Z-score normalization)
    # (x - mean) / std
    gt = (gt - np.mean(gt))
    pred = (pred - np.mean(pred))

    std_gt = np.std(gt)
    std_pred = np.std(pred)

    eps = np.finfo(float).eps

    # 计算协方差 / (std * std)
    # 也就是 Pearson 公式
    cc = np.sum(gt * pred) / (len(gt) * std_gt * std_pred + eps)

    return cc


def cal_kld(gt, pred):
    """
    计算 KL 散度 (Kullback-Leibler Divergence, KLD)
    衡量分布差异，值范围 [0, +inf)，越接近 0 越好。

    公式: sum( GT * log(GT / Pred + eps) )
    注意: KLD 是非对称的，通常用 GT 作为参考分布。
    """
    # 1. 必须先进行概率密度归一化 (Sum = 1)
    gt = normalize_density(gt)
    pred = normalize_density(pred)

    # 2. 加上极小值防止 log(0) 或 除以 0
    eps = np.finfo(float).eps

    # 3. 计算 KLD
    # 只有当 GT > 0 的地方才计算，避免 log(0)
    # 但加上 eps 后可以直接计算
    kld = np.sum(gt * np.log((gt + eps) / (pred + eps)))

    return kld
