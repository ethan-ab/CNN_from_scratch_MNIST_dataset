from torch.utils.data import random_split
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms


train_path = '../../data'
test_path = '../../data'

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_set = datasets.MNIST(root=train_path, train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=test_path, train=False, download=True,transform=transform)

val_size = int(0.2 * len(train_set))

train_set, val_set = random_split(train_set, [len(train_set) - val_size, val_size])
