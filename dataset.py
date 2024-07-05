from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader, Subset

def get_data_loader(dataset_name, batch_size):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)
    ])
    ds_fn = FashionMNIST if dataset_name == "fashion" else MNIST
    dataset = ds_fn("../../../datasets", download=True, transform=transform)
    # dataset = Subset(dataset, indices=range(200))  # 选取部分数据用于测试
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader