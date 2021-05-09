import torch


class SeriesDataset:
    def __init__(self):
        pass

    def getSyntheticData(self, size, length):
        # return sequences of length, size times. size is the number of rows.
        return torch.randint(0, 10, (size, length))

    def toHotVec(self, tensor):
        return torch.nn.functional.one_hot(tensor,10).float()

    def getSyntheticDataInHotVector(self, size, batch, length):
        # return one hot vector
        batches = self.getSyntheticData(size, length).split(batch)
        return list(map(self.toHotVec, batches))
