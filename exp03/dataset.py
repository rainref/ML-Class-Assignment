import os

from PIL import Image
from torch.utils.data import Dataset

TRAIN_PATH = "3-Saliency-TrainSet"
TEST_PATH = "3-Saliency-TestSet"


class SaliencyDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        root_dir: 数据集根目录 (例如 '3-Saliency-TrainSet')
        结构假设: root_dir 下包含 'Stimuli' 和 'FIXATIONMAPS'，且各自内部有分类文件夹
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths = []
        self.label_paths = []
        self.categories = []

        # 遍历目录结构
        stimuli_root = os.path.join(root_dir, 'Stimuli')
        fix_root = os.path.join(root_dir, 'FIXATIONMAPS')

        # 获取所有类别文件夹
        categories = [d for d in os.listdir(stimuli_root) if os.path.isdir(os.path.join(stimuli_root, d))]
        categories.sort()

        for cat in categories:
            cat_stim_path = os.path.join(stimuli_root, cat)
            cat_fix_path = os.path.join(fix_root, cat)

            # 获取该类别下的所有图片
            imgs = [f for f in os.listdir(cat_stim_path) if f.lower().endswith('.jpg')]

            for img_name in imgs:
                self.image_paths.append(os.path.join(cat_stim_path, img_name))
                self.label_paths.append(os.path.join(cat_fix_path, img_name))
                self.categories.append(cat)  # 记录类别用于后续分析

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取彩色图像
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # 读取显著图（灰度）
        label = Image.open(self.label_paths[idx]).convert('L')

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label, self.categories[idx], self.image_paths[idx]
