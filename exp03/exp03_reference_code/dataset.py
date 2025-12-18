import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SaliencyDataset(Dataset):
    def __init__(self, root_dir, img_size=(256, 256), is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.is_train = is_train

        # 递归获取所有图像路径
        self.img_paths = []
        img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
        for root, _, files in os.walk(os.path.join(root_dir, "Stimuli")):
            for file in files:
                if file.lower().endswith(img_extensions):
                    self.img_paths.append(os.path.join(root, file))

        # 匹配掩码路径
        self.mask_paths = []
        for img_path in self.img_paths:
            mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")
            mask_path = os.path.splitext(mask_path)[0]
            found = False
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                candidate = mask_path + ext
                if os.path.exists(candidate):
                    self.mask_paths.append(candidate)
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"未找到{img_path}对应的掩码文件")

        # 数据增强
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
        ]) if is_train else None

        print(f"成功加载{len(self.img_paths)}个样本")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 读取原图并记录尺寸
        img_path = self.img_paths[idx]
        img_ori = cv2.imread(img_path)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = img_ori.shape[:2]  # 保存原图尺寸

        # 预处理输入图像（resize到256*256）
        img = cv2.resize(img_ori, self.img_size)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 读取并预处理掩码
        mask_path = self.mask_paths[idx]
        mask_ori = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask_ori, self.img_size)
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # 数据增强
        if self.transform and self.is_train:
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)
            img = self.transform(img)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # 返回原图尺寸和原始掩码（用于测试指标计算）
        return img, mask, (ori_h, ori_w), mask_ori, img_ori
