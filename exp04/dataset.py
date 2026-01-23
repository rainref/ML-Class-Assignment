from torchvision import transforms
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root='./CIFARdata', download=True, transform=transforms.ToTensor())
