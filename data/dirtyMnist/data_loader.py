# gpu
import torch
import ddu_dirty_mnist
from torchvision import datasets, transforms
def load_dataset_dmnist():
    mnist_mean, mnist_std = 0.1307, 0.3081

    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((mnist_mean,), (mnist_std,))])

    dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST("./data/dirtyMnist/data", train=True,transform=None, download=True, device="cuda")
    dirty_mnist_test = ddu_dirty_mnist.DirtyMNIST("./data/dirtyMnist/data", train=False,transform=None, download=True, device="cuda")
    

    return dirty_mnist_train,dirty_mnist_test
# len(dirty_mnist_train), len(dirty_mnist_test)

def load_dataset_dmnist224():
    mnist_mean, mnist_std = 0.1307, 0.3081

    transform = transforms.Compose([

        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        # transforms.Grayscale(num_output_channels=3),
        # transforms.ToTensor()
    ])

    dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST("./data/dirtyMnist/data", train=True, transform=transform, download=True,
                                                   device="cuda")
    dirty_mnist_test = ddu_dirty_mnist.DirtyMNIST("./data/dirtyMnist/data", train=False, transform=transform, download=True,
                                                  device="cuda")

    return dirty_mnist_train, dirty_mnist_test

def load_dataset_mnist224():
    mnist_mean, mnist_std = 0.1307, 0.3081

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为224x224
        # transforms.ToTensor(),
        # transforms.Normalize((mnist_mean,), (mnist_std,)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.1307,),(0.3081,))
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mnist_train_dataset = datasets.MNIST(root=".data/dirtyMnist/mnist_data/", train=True, transform=transform, download=True)
    mnist_test_dataset = datasets.MNIST(root=".data/dirtyMnist/mnist_data/", train=False, transform=transform, download=True)

    return mnist_train_dataset,mnist_test_dataset

def load_dataset_mnist():
    mnist_mean, mnist_std = 0.1307, 0.3081

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mnist_mean,), (mnist_std,))])

    mnist_train_dataset = datasets.MNIST(root=".data/dirtyMnist/mnist_data/", train=True, transform=transform, download=True)
    mnist_test_dataset = datasets.MNIST(root=".data/dirtyMnist/mnist_data/", train=False, transform=transform, download=True)

    return mnist_train_dataset,mnist_test_dataset

# gpu
def data_loader(dirty_mnist_train,dirty_mnist_test,batch_size):
    dirty_mnist_train_dataloader = torch.utils.data.DataLoader(
        dirty_mnist_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    dirty_mnist_test_dataloader = torch.utils.data.DataLoader(
        dirty_mnist_test,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dirty_mnist_train_dataloader,dirty_mnist_test_dataloader


