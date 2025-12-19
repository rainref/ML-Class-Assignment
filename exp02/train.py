import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from densenet import densenet121
from draw_roc import draw_roc_pr_curves, plot_loss_curve


class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        医学图像数据集类
        Args:
            data_dir: 数据目录路径
            transform: 数据预处理变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # 读取所有图像文件
        for filename in os.listdir(data_dir):
            # 解析文件名：标签_序号.jpg
            # 0是normal，1是disease
            label_str, _ = filename.split('_')
            label = 0 if label_str == "normal" else 1
            filepath = os.path.join(data_dir, filename)
            self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图像
        image = Image.open(img_path).convert('RGB')  # 转换为RGB图

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 改为 224
    transforms.ToTensor(),
    # ... 其他增强 ...
    # 使用标准的 ImageNet 归一化参数，配合预训练模型效果最好
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 数据预处理
# transform = transforms.Compose([
#     transforms.Resize((600, 600)),  # 调整图像尺寸以适应任务输入要求
#     transforms.ToTensor(),  # 转换为Tensor
#
#     # 2. 随机翻转：
#     # 水平翻转 (p=0.5)：大多数医学图像（肺、脑、骨骼）允许左右镜像。
#     transforms.RandomHorizontalFlip(p=0.5),
#     # 垂直翻转 (p=0.5)：注意！如果是全身X光或胸片，建议注释掉这行；如果是病理切片/细胞图，建议开启。
#     transforms.RandomVerticalFlip(p=0.5),
#
#     # 3. 随机旋转：
#     # 旋转 +/- 15度。模拟拍摄时没摆正的情况。不建议超过30度，否则黑边太多。
#     transforms.RandomRotation(degrees=15),
#
#     # 4. 仿射变换 (平移、缩放、剪切)：
#     # translate: 随机平移 10% 的距离
#     # scale: 随机缩放 0.9 到 1.1 倍
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#
#     # 5. 颜色抖动 (ColorJitter)：
#     # brightness(亮度): 模拟X光机/CT电流强弱
#     # contrast(对比度): 模拟组织显影差异
#     # saturation/hue: 对于灰度图(CT/X光)影响不大，但对病理图(Pathology)非常重要
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
#
#     # 6. (可选) 高斯模糊：
#     # 模拟某些图像对焦不准或运动模糊的情况
#     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
#
#     transforms.Normalize([0.5, 0.5, 0.5], [0.229, 0.224, 0.225])  # 使用ImageNet的均值和标准差进行归一化
# ])

# Mean: tensor([0.5311, 0.3296, 0.2173])
# Std: tensor([0.1393, 0.1013, 0.0651])

# 加载数据集
def load_medical_image_dataset(data_dir, batch_size=16, shuffle=True):
    """
    加载医学图像数据集
    Args:
        data_dir: 数据目录路径
        batch_size: 批量大小
        shuffle: 是否打乱数据
    Returns:
        DataLoader对象
    """
    dataset = MedicalImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_model(num_epochs, batch_size, learning_rate, checkpoint_path='./densenet121_medical_image_checkpoint.pth'):
    # 训练设备的选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Using device:", device)
    # 构建ResNet-50模型
    net = densenet121(num_classes=2)
    net = net.to(device)
    # 打印模型结构
    print(net)
    # 加载数据集
    train_loader = load_medical_image_dataset("../../autodl-tmp/2-MedImage-TrainSet/", batch_size=batch_size,
                                              shuffle=True)
    test_loader = load_medical_image_dataset("../../autodl-tmp/2-MedImage-TestSet/", batch_size=batch_size,
                                             shuffle=False)
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 'min'表示指标下降时衰减，'max'表示上升时衰减
        factor=0.5,  # 衰减系数
        patience=5,  # 容忍多少个epoch指标不改善
        min_lr=1e-6  # 最小学习率
    )
    # 初始化训练参数
    test_accuracies = []
    loss_values = []
    best_val_loss = float('inf')

    # 检查是否存在权重文件，如果存在则加载
    if os.path.exists(checkpoint_path):
        print(f"检测到权重文件: {checkpoint_path}")
        print("正在加载权重并恢复训练状态...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 加载模型权重
        net.load_state_dict(checkpoint)


    else:
        print("未找到权重文件，开始新的训练...")

    # 训练模型
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                loss_values.append(running_loss / 10)
                running_loss = 0.0

        # 每个epoch结束后，在测试集上评估模型
        net.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(test_loader.dataset)
        # 更新学习率
        scheduler.step(avg_val_loss)
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), './densenet121_medical_image_best_{}.pth'.format(epoch + 1))
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")

        if epoch % 10 == 9:
            # 每10个epoch保存一次模型权重
            torch.save(net.state_dict(), './densenet121_medical_image_{}.pth'.format(epoch + 1))
            print(f'Checkpoint saved at epoch {epoch + 1}')

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Accuracy on test set after epoch {epoch + 1}: {accuracy:.2f}%')

    # 保存模型参数
    torch.save(net.state_dict(), './densenet121_medical_image_final.pth')
    print('Training Finished')
    # 绘制损失曲线
    plot_loss_curve(loss_values)


def evaluate_model(model_path, test_data_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = densenet121(num_classes=2)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()

    test_loader = load_medical_image_dataset(test_data_dir, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    y_pred_prob = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1)
            y_pred_prob.extend(probabilities.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    draw_roc_pr_curves(all_labels, y_pred_prob, all_predictions)


if __name__ == "__main__":
    train_model(num_epochs=50, batch_size=16, learning_rate=0.001, checkpoint_path='./xx.pth')
    evaluate_model('densenet121_medical_image.pth', '../../autodl-tmp/2-MedImage-TestSet/', batch_size=16)
