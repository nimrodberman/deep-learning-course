import numpy as np
from work1.task1 import Functions
from work1.task1.GradientTest import *
from scipy.io import loadmat

if __name__ == '__main__':
    print("main started")

    # upload data and set parameters for it
    sample_size = 20
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
    trainSetX_batches = np.array_split(trainSetX, 3000, axis=1)
    trainSetY_batches = np.array_split(trainSetY, 3000, axis=1)

    # set parameters
    output_size = len(trainSetY_batches[0][:, 0])
    input_size = trainSetX_batches[0].shape[0]
    theta = np.random.rand(input_size, output_size)
    bias = np.zeros([1, output_size])

    # gradient test
    WeightsGradientTest(theta,bias,trainSetX_batches[0],trainSetY_batches[0])


    iterations = 1000
    learningRate = 0.0001
    for i in range(0, iterations):
        for batchX, batchY in zip(trainSetX_batches, trainSetY_batches):
            loss, grad_theta, grad_b, _ = softmaxRegression(theta, batchX, bias, batchY)
            theta = theta - (learningRate * grad_theta)
            bias = bias - (learningRate * grad_b)

        if i % 10 == 0:
            _, _, _, pred = softmaxRegression(theta, testSetX, bias, testSetY)
            acc = accuracy(pred, testSetY)
            print(f"iter number:{i}/{iterations}   loss: {loss}    accuracy:{acc}%")
