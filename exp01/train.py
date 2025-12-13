import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from resnet import resnet50


# 读取TrainingSet中的数据集

class MNISTBMPDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        MNIST BMP数据集类
        Args:
            data_dir: 数据目录路径
            transform: 数据预处理变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # 读取所有BMP文件
        for filename in os.listdir(data_dir):
            if filename.endswith('.bmp'):
                # 解析文件名：标签_序号.bmp
                label_str, _ = filename.split('_')
                label = int(label_str)
                filepath = os.path.join(data_dir, filename)
                self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图像
        image = Image.open(img_path).convert('L')  # 转换为灰度图

        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 确保尺寸一致
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1,1]
])


# 加载数据集
def load_mnist_bmp_dataset(data_dir, batch_size=32, shuffle=True):
    """
    加载MNIST BMP数据集
    """
    dataset = MNISTBMPDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {len(set([label for _, label in dataset.samples]))}")

    return dataloader, dataset


# 画出损失曲线
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
    #plt.show()
    plt.savefig('./training_loss_curve.png')

    # 打印统计信息
    #print(f"总训练步数: {len(loss_values)}")
    print(f"最终Loss: {final_loss:.4f}")
    print(f"最小Loss: {min_loss:.4f}")
    print(f"平均Loss: {mean_loss:.4f}")



def train_model(num_epochs, batch_size):
    '''
    训练模型函数
    Args:
        num_epochs: 训练轮数
        batch_size: 批次大小
    '''
    # 训练设备的选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Using device:", device)
    # 构建ResNet-50模型
    net = resnet50(num_classes=10, include_top=True)
    net = net.to(device)

    # 打印模型结构
    print(net)
    # 加载MNIST BMP数据集
    train_loader, train_dataset = load_mnist_bmp_dataset('./TrainingSet', batch_size=batch_size)
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_values = []
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

            loss_values.append(loss.item())

            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    # 保存模型参数
    torch.save(net.state_dict(), 'resnet50_mnist_bmp.pth')
    plot_loss_curve(loss_values)
    print('Training Finished')



# 验证模型错误率
def evaluate_model(model_path, test_data_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50(num_classes=10, include_top=True)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()

    test_loader, test_dataset = load_mnist_bmp_dataset(test_data_dir, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')


if __name__ == '__main__':
    num_epochs = 5
    batch_size = 64
    train_model(num_epochs, batch_size)
    print('----Evaluating on Test Set----')
    evaluate_model('resnet50_mnist_bmp.pth', './TestSet', batch_size)
    print('----Evaluating on Training Set----')
    evaluate_model('resnet50_mnist_bmp.pth', './TrainingSet', batch_size)

