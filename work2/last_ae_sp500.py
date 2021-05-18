from work2.datasets import *
import numpy as np
import matplotlib.pyplot as plt


def printSequenceOnGraph(seq,stockName):
    x = range(len(seq))
    y = seq
    plt.plot(x, y, label='{}'.format(stockName))


def printMultipleSequencesOnGraph(sequences, stockNames, title):
    for seq, name in zip(sequences, stockNames):
        printSequenceOnGraph(seq, name)
    plt.xlabel('time steps')
    plt.ylabel('signal value')
    plt.title(title)
    plt.legend()
    plt.savefig('{}.png'.format(title))
    plt.show()


# ----- 3.3.1 daily maximum ----#
sp500data = sp500Dataset()
# amazonData = sp500data.getAllCompanyData('AMZN')
# googleData = sp500data.getAllCompanyData('GOOGL')
# amazonHigh = amazonData.filter(items=['high']).to_numpy().T
# googleHigh = googleData.filter(items=['high']).to_numpy().T
# amazonAndGoogleHigh = np.concatenate([amazonHigh, googleHigh])
# printMultipleSequencesOnGraph(amazonAndGoogleHigh, ['AMZM', 'GOOGL'], 'Amazon and Google High Stock Price - 2014 - 2017')
#

# ----- 3.3.2 stocks reconstruction objective ----- #
data = sp500data.getDataForModel()
print("hey")
