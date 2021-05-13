import torch
import pandas as pd


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


class sp500Dataset:
    def __init__(self):
        self.data = pd.read_csv('SP_500_Stock_Prices_2014-2017.csv')

    def getAllData(self):
        return self.data

    def getAllCompanyData(self, companyName):
        return self.data[(self.data.symbol == companyName)]

    def getDataForModel(self):
        # convert to pandas date object
        self.data['date'] = pd.to_datetime(self.data.date)
        # sort by companies name and date
        sortedCompaniesDict = dict(tuple(self.data.sort_values(['symbol', 'date']).groupby('symbol')))
        sortedCompaniesList = list(sortedCompaniesDict.values())
        # filter companies with less then 1007 days of data
        filteredCompanies = list(filter(lambda company: company.shape[0] == 1007, sortedCompaniesList))
        filteredCompanies = list(map(lambda company: torch.tensor(company.to_numpy()[:, 2:6]), filteredCompanies))

        return torch.tensor(filteredCompanies)
