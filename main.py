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

    #  shuffle
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

    # gradient tests TODO - add
    # GradientTest(trainSetX_batches[0], trainSetY_batches[0], theta, bias)
    # gradientTest1(batch_x, theta, b_l, batch_y)
    # Bias_Grad_Test(batch_x, batch_y, theta, b_l)

    # iterations = 1000
    # learningRate = 1e-5
    # losses = []
    # for i in range(0, iterations):
    #     loss, grad_theta, grad_b = Functions.softmaxRegression(theta, batch_x, b_l, batch_y)
    #     losses.append(loss)
    #     theta = theta - (learningRate * grad_theta)
    #     grad_b = grad_b - (learningRate * grad_b)
    #     if i % 10 == 0:
    #         print(f"iter number:{i}/{iterations}   loss: {loss}")
