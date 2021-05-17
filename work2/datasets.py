import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class SeriesDataset:
    def __init__(self):
        pass

    def getSyntheticData(self, size, length):
        # return sequences of length, size times. size is the number of rows.
        return torch.randint(0, 10, (size, length))  # TODO should be int or rational??

    def toHotVec(self, tensor):
        return torch.nn.functional.one_hot(tensor, 10).float()

    def getSyntheticDataInHotVector(self, size, batch, length):
        # return one hot vector
        batches = self.getSyntheticData(size, length).split(batch)
        return list(map(self.toHotVec, batches))


def getMnistDataLoader(batch_size,test_size):
    # For normal distribution
    mean = 0
    std = 1
    # Set data transformation
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor()])
    # Download the data if needed
    train_set = MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=test_transform)
    # Data shape = (num_of_batch X batch_size X T X input_size)

    # Set data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_set, batch_size=test_size, shuffle=True)

    # Return data iterator
    return train_loader, test_loader

#
# if __name__ == '__main__':
#     # x = getMnistData(8)
#     mean = 0
#     std = 1
#     train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
#     train_set = MNIST(root='./data', train=True, download=True, transform=train_transform)
#     loader = DataLoader(train_set, batch_size=1, shuffle=False)
#     image, label = next(iter(loader))
#     len(train_set)
#     x = 5
