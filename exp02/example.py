import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_model_with_resume(net, train_loader, val_loader, num_epochs=100,
                            checkpoint_path='densenet121_medical_image.pth',
                            device='cuda'):
    """
    带有断点续训功能的训练函数

    参数:
    - net: 模型实例
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - num_epochs: 总训练轮数
    - checkpoint_path: 权重文件路径
    - device: 训练设备 ('cuda' 或 'cpu')
    """

    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # 初始化训练参数
    start_epoch = 0
    best_val_loss = float('inf')
    train_history = []
    val_history = []

    # 检查是否存在权重文件，如果存在则加载
    if os.path.exists(checkpoint_path):
        print(f"检测到权重文件: {checkpoint_path}")
        print("正在加载权重并恢复训练状态...")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型权重
        net.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载学习率调度器状态
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载训练历史
        if 'train_history' in checkpoint:
            train_history = checkpoint['train_history']

        if 'val_history' in checkpoint:
            val_history = checkpoint['val_history']

        # 加载最佳验证损失
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']

        # 设置起始epoch
        start_epoch = checkpoint['epoch']

        # 加载学习率（如果有）
        if 'learning_rate' in checkpoint:
            for param_group in optimizer.param_groups:
                param_group['lr'] = checkpoint['learning_rate']

        print(f"从第 {start_epoch + 1} 个epoch继续训练...")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    else:
        print("未找到权重文件，开始新的训练...")

    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 打印进度
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total

        # 验证阶段
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存训练历史
        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_history,
                'val_history': val_history,
                'best_val_loss': best_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, checkpoint_path.replace('.pth', '_best.pth'))
            print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")

        # 定期保存检查点（每5个epoch）
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_history,
                'val_history': val_history,
                'best_val_loss': best_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, checkpoint_path.replace('.pth', f'_epoch_{epoch + 1}.pth'))

        # 总是保存最新的检查点（用于断点续训）
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_history': train_history,
            'val_history': val_history,
            'best_val_loss': best_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, checkpoint_path)

        # 打印epoch统计信息
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n{'=' * 60}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.1f}s")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"{'=' * 60}\n")

    print("训练完成！")

    # 返回训练历史供可视化使用
    return {
        'train_loss': train_history,
        'val_loss': val_history,
        'best_val_loss': best_val_loss
    }


# 使用示例
if __name__ == "__main__":
    # 假设你已经定义了模型和数据加载器
    # net = YourDenseNetModel()
    # train_loader, val_loader = get_data_loaders()

    # 训练模型（自动检测并加载已有权重）
    history = train_model_with_resume(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        checkpoint_path='checkpoints/densenet121_medical_image.pth',
        device='cuda'
    )
