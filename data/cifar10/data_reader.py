import torchvision
import torchvision.transforms as transforms
import torch

DOWNLOAD_CIFAR10 = True

def load_dataset():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_data = torchvision.datasets.CIFAR10(
        root='./data/cifar10',
        train=True,
        download=DOWNLOAD_CIFAR10,
        transform=transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./data/cifar10',
        train=False,
        download=DOWNLOAD_CIFAR10,
        transform=transform,
    )

    return train_data, test_data

def load_dataset_cifar10():
    norm_mean = [0.485, 0.456, 0.406]  # 均值
    norm_std = [0.229, 0.224, 0.225]  # 方差
    transform_train = transforms.Compose([transforms.ToTensor(),  # 将PILImage转换为张量
                                          # 将[0,1]归一化到[-1,1]
                                          transforms.Normalize(norm_mean, norm_std),
                                          transforms.RandomHorizontalFlip(),  # 随机水平镜像
                                          transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
                                          transforms.RandomCrop(32, padding=4)  # 随机中心裁剪
                                          ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(norm_mean, norm_std)])

    train_data = torchvision.datasets.CIFAR10(
        root='./data/cifar10',
        train=True,
        download=DOWNLOAD_CIFAR10,
        transform=transform_train
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./data/cifar10',
        train=False,
        download=DOWNLOAD_CIFAR10,
        transform=transform_test,
    )

    return train_data, test_data

def sample_data(dataset_train,dataset_test):
    sample_index_train = [i for i in range(len(dataset_train))]  # 假设取
    X_train = []
    y_train = []
    for i in sample_index_train:
        X = dataset_train[i][0]
        X_train.append(X)
        y = dataset_train[i][1]
        y_train.append(y)
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)

    sample_index_test = [i for i in range(len(dataset_test))]
    X_test = []
    y_test = []
    for i in sample_index_test:
        X = dataset_test[i][0]
        X_test.append(X)
        y = dataset_test[i][1]
        y_test.append(y)
    X_test = torch.cat(X_test, dim=0)
    y_test = torch.cat(y_test, dim=0)

    # sampled_train_data = [(X, y) for X, y in zip(X_train, y_train)]  # 包装为数据对
    return X_train, y_train, X_test, y_test