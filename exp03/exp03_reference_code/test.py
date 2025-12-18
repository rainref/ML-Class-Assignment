import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SaliencyDataset
from metric import calc_cc_score, KLD
from model import ResNet18Saliency


@torch.no_grad()
def test_and_evaluate(model, test_dir, save_dir="saliency_results", img_size=(256, 256)):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    # 加载测试集（获取原图尺寸和原始掩码）
    test_dataset = SaliencyDataset(test_dir, img_size=img_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化指标存储
    all_cc = []
    all_kl = []

    pbar = tqdm(test_loader, desc="测试与评估")
    for idx, (img, _, (ori_h, ori_w), mask_ori, img_ori) in enumerate(pbar):
        category = os.path.basename(os.path.dirname(test_dataset.img_paths[idx]))
        cate_save_dir = os.path.join(save_dir, category)
        os.makedirs(cate_save_dir, exist_ok=True)
        # 模型预测（256*256）
        img = img.to(device)
        saliency_pred = model(img).squeeze().cpu().numpy()  # [256, 256]

        # 将预测结果resize回原图尺寸
        saliency_pred_ori = cv2.resize(saliency_pred, (ori_w.item(), ori_h.item()))  # [ori_h, ori_w]
        mask_ori = mask_ori.squeeze().cpu().numpy()

        # 计算指标
        cc_score = calc_cc_score(mask_ori, saliency_pred_ori)
        kl_score = KLD(mask_ori, saliency_pred_ori)
        all_cc.append(cc_score)
        all_kl.append(kl_score)

        # 保存结果
        img_name = os.path.splitext(os.path.basename(test_dataset.img_paths[idx]))[0]

        # 保存resize回原尺寸的显著性图
        saliency_pred_save = (saliency_pred_ori * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(cate_save_dir, f"{img_name}.png"), saliency_pred_save)

    # 计算平均指标
    avg_cc = np.mean(all_cc)
    avg_kl = np.mean(all_kl)
    print(f"\n测试集平均CC系数：{avg_cc:.4f}")
    print(f"测试集平均KL散度：{avg_kl:.4f}")

    # 保存指标结果
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"平均CC系数：{avg_cc:.4f}\n")
        f.write(f"平均KL散度：{avg_kl:.4f}\n")
        f.write(f"所有CC值：{all_cc}\n")
        f.write(f"所有KL值：{all_kl}\n")

    return avg_cc, avg_kl


# ===================== 主函数 =====================
if __name__ == "__main__":
    TEST_DIR = "3-Saliency-TestSet"  # 测试集根目录（需与训练集结构一致）

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "resnet18_saliency_best.pth"  # 最佳模型保存路径

    # 初始化模型、损失函数、优化器
    model = ResNet18Saliency(pretrained=True).to(DEVICE)

    # 加载最佳模型并测试
    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n加载最佳模型（Epoch {checkpoint['epoch'] + 1}）")
    avg_cc, avg_kl = test_and_evaluate(model, TEST_DIR, save_dir="saliency_results")
    print("测试完成！结果已保存至 saliency_results 目录")
