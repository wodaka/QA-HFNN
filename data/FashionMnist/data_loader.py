import torch
from torchvision import datasets, transforms

def load_dataset_fashionmnist():
    mnist_mean, mnist_std = 0.1307, 0.3081

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mnist_mean,), (mnist_std,))])

    fashionmnist_train_dataset = datasets.FashionMNIST(root=".data/FashionMnist/data/", train=True, transform=transform,
                                         download=True)
    fashionmnist_test_dataset = datasets.FashionMNIST(root=".data/FashionMnist/data/", train=False, transform=transform,
                                        download=True)

    return fashionmnist_train_dataset, fashionmnist_test_dataset