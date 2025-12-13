import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, f1_score

from resnet import resnet50
from train import load_mnist_bmp_dataset


# 评估模型在测试集上的精确率、召回率、F1分数
def evaluate_model(model_path, test_data_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50(num_classes=10, include_top=True)
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net.eval()

    test_loader, test_dataset = load_mnist_bmp_dataset(test_data_dir, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    print(f'Recall of the model on the test set: {recall:.4f}')
    print(f'F1 Score of the model on the test set: {f1:.4f}')
    plot_confusion_matrix(all_labels, all_predictions)

# 画出混淆矩阵
def plot_confusion_matrix(all_labels, all_predictions, class_names=[0,1,2,3,4,5,6,7,8,9]):
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    #plt.show()
    plt.savefig('./confusion_matrix.png')

if __name__ == '__main__':
    model_path = 'resnet50_mnist_bmp.pth'
    test_data_dir = './TestSet'
    batch_size = 64
    evaluate_model(model_path, test_data_dir, batch_size)