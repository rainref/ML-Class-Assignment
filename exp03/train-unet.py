import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import SaliencyDataset
from loss import SaliencyLoss
from unet import UNet
from utils import plot_loss_curve

# 配置路径
TRAIN_PATH = "3-Saliency-TrainSet"
TEST_PATH = "3-Saliency-TestSet"

# 定义预处理
# 输入图像标准化，缩放到模型输入大小 (如 256x256)
input_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 标签仅缩放并转为Tensor
target_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def train(num_epochs, batch_size, learning_rate, pretrained_model=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loss_values = []

    # 1. 准备数据
    train_dataset = SaliencyDataset(TRAIN_PATH, transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 2. 初始化模型
    model = UNet(3, 1).to(device)
    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))
        print("Loaded pretrained model:", pretrained_model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SaliencyLoss(weight_kld=2.0, weight_cc=1.0, weight_bce=1.0)

    # 3. 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for imgs, labels, _, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            # loss = criterion(outputs, labels)
            loss, bce, cc, kld = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            loss_values.append(running_loss / (pbar.n + 1))

        # 保存模型
        torch.save(model.state_dict(), f"saliency_model-{epoch + 21}.pth")
    plot_loss_curve(loss_values, "training_loss_curve.png")
    print("Training finished.")


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="validation")
    for imgs, masks, _, _ in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 如果是主程序则运行训练
if __name__ == '__main__':
    # 确保文件夹存在后再运行，这里假设数据已解压
    if os.path.exists(TRAIN_PATH):
        train(30, batch_size=16, learning_rate=1e-4, pretrained_model='pretrainMLE+sota-loss-20/saliency_model-19.pth')
