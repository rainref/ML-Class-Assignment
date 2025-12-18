import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SaliencyDataset
from metric import cal_cc as cc_metric
from metric import cal_kld as kld_metric
from metric import normalize_map
from unet import UNet

TEST_PATH = "3-Saliency-TestSet"
input_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 标签仅转为Tensor
target_transform = transforms.Compose([
    transforms.ToTensor()
])


def evaluate(model):
    # 测试集不进行 Target Resize，因为我们需要和原始尺寸的 GT 比较
    test_dataset = SaliencyDataset(TEST_PATH, transform=input_transform, target_transform=target_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 存储结果
    results_by_cat = {}

    # 创建输出文件夹
    output_dir = "Results_Resnet"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (img_tensor, gt_raw, cat, img_path) in enumerate(tqdm(test_loader, desc="Evaluating")):
            img_tensor = img_tensor.to(device)
            cat = cat[0]  # batch size 为 1

            # 预测
            pred_map = model(img_tensor)

            # 后处理：转换回 numpy，并 resize 到原始图像大小
            # gt_raw 是 tensor 但没有 resize，保留了原始尺寸 (Batch, 1, H, W)
            orig_h, orig_w = gt_raw.shape[2], gt_raw.shape[3]

            pred_np = pred_map.squeeze().cpu().numpy()
            pred_resized = cv2.resize(pred_np, (orig_w, orig_h))

            # 读取原始 GT 用于计算指标 (转为 numpy)
            gt_np = gt_raw.squeeze().cpu().numpy()

            # 归一化用于计算指标
            pred_norm = normalize_map(pred_resized)
            gt_norm = normalize_map(gt_np)

            # 计算指标
            cc = cc_metric(gt_norm, pred_norm)
            kld = kld_metric(gt_norm, pred_norm)

            # 记录数据
            if cat not in results_by_cat:
                results_by_cat[cat] = {'cc': [], 'kld': []}
            results_by_cat[cat]['cc'].append(cc)
            results_by_cat[cat]['kld'].append(kld)

            # 主观结果保存 (可选，保存前1-2张每类，或者全部)
            # 保存为热力图覆盖在原图上，或者单张灰度图
            save_name = os.path.basename(img_path[0])
            save_cat_dir = os.path.join(output_dir, cat)
            os.makedirs(save_cat_dir, exist_ok=True)

            # 保存预测的灰度图
            pred_img = (pred_resized * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_cat_dir, save_name), pred_img)

    # 输出总体和分类统计结果
    print("\n" + "=" * 40)
    print(f"{'Category':<20} | {'CC (Mean)':<10} | {'KLD (Mean)':<10}")
    print("-" * 45)

    total_cc = []
    total_kld = []

    for cat, metrics in results_by_cat.items():
        mean_cc = np.nanmean(metrics['cc'])
        mean_kld = np.nanmean(metrics['kld'])
        total_cc.extend(metrics['cc'])
        total_kld.extend(metrics['kld'])

        print(f"{cat:<20} | {mean_cc:.4f}     | {mean_kld:.4f}")

    print("-" * 45)
    print(f"{'OVERALL':<20} | {np.mean(total_cc):.4f}     | {np.mean(total_kld):.4f}")
    print("=" * 40)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================评估老师的参考答案================#
    # checkpoint_path="exp03_reference_code/resnet18_saliency_best.pth"
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model = ResNet18Saliency(pretrained=False).to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    model = UNet(3, 1).to(device)
    model.load_state_dict(torch.load('pretrainMLE+sota-loss-20/saliency_model-19.pth'))

    model.eval()

    evaluate(model=model)
