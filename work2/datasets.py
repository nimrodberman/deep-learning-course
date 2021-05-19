import torch
import torch.nn.functional as f
import torchvision
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np


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


class sp500Dataset:
    def __init__(self):
        self.data = pd.read_csv('SP_500_Stock_Prices_2014-2017.csv')

    def getAllData(self):
        return self.data

    def getAllCompanyData(self, companyName):
        return self.data[(self.data.symbol == companyName)]

    def getDataForModel(self, batch_size, time=1007, show_data=False):
        unnormalize_data = self.data.copy()
        # convert to pandas date object
        self.data['date'] = pd.to_datetime(self.data.date)
        # normalize data
        x = self.data.drop(['date', 'symbol'], axis=1).values

        # x = self.data.drop(['date', 'symbol', 'open', 'close', 'low', 'volume'], axis=1).values

        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        self.data[['open', 'high', 'low', 'close', 'volume']] = x_scaled

        # self.data['high'] = x_scaled

        # sort by companies name and date
        sortedCompaniesDict = dict(tuple(self.data.sort_values(['symbol', 'date']).groupby('symbol')))
        sortedCompaniesList = list(sortedCompaniesDict.values())

        # filter companies with less then 1007 days of data
        filteredCompanies = list(filter(lambda company: company.shape[0] == 1007, sortedCompaniesList))

        # keep only high
        filteredCompanies = list(
            map(lambda company: torch.from_numpy(company.to_numpy()[:, 3].astype(float))[:time], filteredCompanies))

        data_tesor = torch.stack(filteredCompanies)

        if show_data:
            self.show_amazon_and_google(unnormalize_data)

        m = len(data_tesor)
        train = data_tesor[:(int)(3 / 4 * m)].float()
        validation = data_tesor[(int)(3 / 4 * m):].float()

        batchs = torch.split(train, batch_size, dim=0)
        return batchs, validation

    def show_amazon_and_google(self, data):
        # amz_idx = 37
        # gogl_idx = 197
        data['date'] = pd.to_datetime(data.date)

        # sort by companies name and date
        sortedCompaniesDict = dict(tuple(data.sort_values(['symbol', 'date']).groupby('symbol')))
        sortedCompaniesList = list(sortedCompaniesDict.values())

        googl_dict = sortedCompaniesDict['GOOGL']
        amzn_dict = sortedCompaniesDict['AMZN']

        google_lst = list(googl_dict.values)
        google_arr = np.array(google_lst)
        googl = np.reshape(google_arr, (1007, 7))[:, 3]

        amzn_lst = list(amzn_dict.values)
        amzn_arr = np.array(amzn_lst)
        amzn = np.reshape(amzn_arr, (1007, 7))[:, 3]

        # sortedCompaniesList = list(sortedCompaniesDict.values())

        # filter companies with less then 1007 days of data
        # filteredCompanies = list(filter(lambda company: company.shape[0] == 1007, sortedCompaniesList))

        # keep only high value
        # high_vals_googl = list(torch.from_numpy(google_lst[:, 3:4].astype(float)))
        date_vals = list(
            map(lambda company: company.to_numpy()[:, 1:2], sortedCompaniesList))[0]

        # data_tesor = torch.stack(high_vals)
        # googl = high_vals[gogl_idx]
        # amzn = high_vals[amz_idx]

        plt.plot(date_vals, googl, label='GOOGL')
        plt.plot(date_vals, amzn, label='AMZN')
        plt.xlabel('date')
        plt.ylabel('high value')
        plt.title('Amazon And Google 2014-2017')
        plt.legend()
        plt.show()
        x = 5

#
# data_gen = sp500Dataset()
# dataset = data_gen.getDataForModel(20, False)
