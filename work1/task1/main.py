import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from work1.task1 import Functions
from work1.task1.GradientTest import *
if __name__ == '__main__':
    print("main started")

    # upload data and set parameters for it
    # sample_size = 20

    number_of_butches = 200
    learningRate = 0.0001
    iterations = 800


    PeaksData = loadmat('PeaksData.mat')
    trainSet = np.array(PeaksData['Yt'])
    trainSetLabels = np.array(PeaksData['Ct'])

    validationSet = np.array(PeaksData['Yv'])
    validationSetLabels = np.array(PeaksData['Cv'])
    testSetSize = 200

    #  shuffle validation set
    idx = np.random.permutation(len(validationSetLabels[0]))
    testSetX = validationSet[:, idx][:,:testSetSize]
    testSetY = validationSetLabels[:, idx][:,:testSetSize]

    #  shuffle training set
    idx = np.random.permutation(len(trainSetLabels[0]))
    trainSetX = trainSet[:, idx]
    trainSetY = trainSetLabels[:, idx]

    # split into batches
    trainSetX_batches = np.array_split(trainSetX, number_of_butches, axis=1)
    trainSetY_batches = np.array_split(trainSetY, number_of_butches, axis=1)

    # set parameters
    output_size = len(trainSetY_batches[0][:, 0])
    input_size = trainSetX_batches[0].shape[0]
    theta = np.random.rand(input_size, output_size)
    bias = np.zeros([1, output_size])

    # gradient test
    # WeightsGradientTest(theta,bias,trainSetX_batches[0],trainSetY_batches[0])

    train_accuracys=[]
    test_accuracys=[]
    losses=[]
    for i in range(0, iterations):
        for batchX, batchY in zip(trainSetX_batches, trainSetY_batches):
            loss, grad_theta, grad_b, pred = softmaxRegression(theta, batchX, bias, batchY)
            theta = theta - (learningRate * grad_theta)
            bias = bias - (learningRate * grad_b)
        # check current accuracy and store it
        loss, _, _, pred = softmaxRegression(theta, trainSetX, bias, trainSetY)
        acc_train = accuracy(pred, trainSetY)
        _, _, _, pred = softmaxRegression(theta, testSetX, bias, testSetY)
        acc_test = accuracy(pred, testSetY)
        train_accuracys.append(acc_train)
        test_accuracys.append(acc_test)
        losses.append(loss)
        if i % 10 == 0:
            print(f"iter number:{i}/{iterations}   loss: {loss}    accuracy:{acc_test}%    batch size:{batchX.shape[1]}")

    xbar=np.arange(len(train_accuracys))
    # plt.plot(xbar, train_accuracys, label="train")
    # plt.plot(xbar, test_accuracys, label="test")
    #
    # # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',
    # #            ncol=2, mode="expand", borderaxespad=0.)
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.title('Success Percentages')
    # plt.legend('train', 'test')
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(xbar, train_accuracys,color='blue', label='train')
    ax.plot(xbar, test_accuracys, color='red', label='test')
    # ax.plot(xbar, losses, 'k:', label='loss')

    legend = ax.legend(loc='lower right', fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.title('Success Percentages')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()