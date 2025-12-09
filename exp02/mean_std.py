import os

import torch
from PIL import Image
from torchvision import transforms


def calculate_mean_std(data_dir):
    means = []
    stds = []
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor()
    ])

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        image = Image.open(filepath).convert('RGB')
        tensor = transform(image)
        means.append(tensor.mean(dim=(1, 2)))
        stds.append(tensor.std(dim=(1, 2)))

    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    return mean, std


data_dir = "./2-MedImage-TrainSet/"
mean, std = calculate_mean_std(data_dir)
print("Mean:", mean)
print("Std:", std)
