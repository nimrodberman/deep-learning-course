import torch
import torch.nn.functional as f
import torchvision
import pandas as pd
from sklearn import preprocessing


class SeriesDataset:
    def __init__(self):
        pass

    def getSyntheticData(self, size, length):
        # return sequences of length, size times. size is the number of rows.
        return torch.randint(0, 10, (size, length))

    def toHotVec(self, tensor):
        return torch.nn.functional.one_hot(tensor, 10).float()

    def getSyntheticDataInHotVector(self, size, batch, length):
        # return one hot vector
        batches = self.getSyntheticData(size, length).split(batch)
        return list(map(self.toHotVec, batches))


class sp500Dataset:
    def __init__(self):
        self.data = pd.read_csv('SP_500_Stock_Prices_2014-2017.csv')

    def getAllData(self):
        return self.data

    def getAllCompanyData(self, companyName):
        return self.data[(self.data.symbol == companyName)]

    def getDataForModel(self,batch_size):
        # convert to pandas date object
        self.data['date'] = pd.to_datetime(self.data.date)
        # normalize data
        x = self.data.drop(['date', 'symbol'], axis=1).values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.data[['open', 'high', 'low','close','volume']] = x_scaled
        # sort by companies name and date
        sortedCompaniesDict = dict(tuple(self.data.sort_values(['symbol', 'date']).groupby('symbol')))
        sortedCompaniesList = list(sortedCompaniesDict.values())
        # filter companies with less then 1007 days of data
        filteredCompanies = list(filter(lambda company: company.shape[0] == 1007, sortedCompaniesList))
        # keep only open, close, high, low, volume
        filteredCompanies = list(
            map(lambda company: torch.from_numpy(company.to_numpy()[:, 2:6].astype(float)), filteredCompanies))

        data_tesor = torch.stack(filteredCompanies)



        return torch.split(data_tesor, batch_size, dim=0)




data = sp500Dataset()
data_model = data.getDataForModel(20)