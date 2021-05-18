import torch


class SeriesDataset:
    def __init__(self):
        pass

    def getSyntheticData(self, size, length):
        # return sequences of length, size times. size is the number of rows.
        return torch.FloatTensor(size, length, 1).uniform_(0, 1)

    def toHotVec(self, tensor):
        return torch.nn.functional.one_hot(tensor, 10).float()

    def getSyntheticDataInBatches(self, size, batch, length):
        # return one hot vector
        return self.getSyntheticData(size, length).split(batch)




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pandas as pd
from sklearn import preprocessing



def DataGenerator(T):
    df = pd.read_csv("data/all_stocks_5yr.csv")
    x = df.drop(['date','Name'],axis=1).values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    x=5
DataGenerator(1200)
