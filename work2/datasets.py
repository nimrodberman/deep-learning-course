import torch
import torch.nn.functional as f
import torchvision
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


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


class sp500Dataset:
    def __init__(self):
        self.data = pd.read_csv('SP_500_Stock_Prices_2014-2017.csv')

    def getAllData(self):
        return self.data

     # normalize each sample separately
    def getDataForModel(self, batch_size, time=1007, show_data=False):
        unnormalize_data = self.data.copy()
        # convert to pandas date object
        self.data['date'] = pd.to_datetime(self.data.date)

        # sort by companies name and date
        sortedCompaniesDict = dict(tuple(self.data.sort_values(['symbol', 'date']).groupby('symbol')))
        sortedCompaniesList = list(sortedCompaniesDict.values())

        # filter companies with less then 1007 days of data
        filteredCompanies = list(filter(lambda company: company.shape[0] == 1007, sortedCompaniesList))

        # keep only high
        filteredCompanies = list(
            map(lambda company: torch.from_numpy(company.to_numpy()[:, 3].astype(float))[:time], filteredCompanies))

        normalized = list(
            map(lambda company:  self.norm_company(company), filteredCompanies))

        data_tesor = torch.stack(filteredCompanies)

        if show_data:
            self.show_amazon_and_google(unnormalize_data)

        m = len(data_tesor)
        train = data_tesor[:(int)(3 / 4 * m)].float()
        validation = data_tesor[(int)(3 / 4 * m):].float()

        # train, validation = train_test_split(data_tesor, shuffle=True, test_size=0.25)

        batchs = torch.split(train, batch_size, dim=0)
        return batchs, validation

    def norm_company(self, company):
        # normalize data
        company -= company.min()
        company /= company.max()

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

        date_vals = list(
            map(lambda company: company.to_numpy()[:, 1:2], sortedCompaniesList))[0]

        plt.plot(date_vals, googl, label='GOOGL')
        plt.plot(date_vals, amzn, label='AMZN')
        plt.xlabel('date')
        plt.ylabel('high value')
        plt.title('Amazon And Google 2014-2017')
        plt.legend()
        plt.show()
        x = 5
