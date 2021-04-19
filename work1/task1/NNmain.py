import numpy as np
from work1.task1 import Functions
from work1.task1.JacAndGradTest import *
from scipy.io import loadmat

if __name__ == '__main__':
    print("main started")

    # upload data and set parameters for it
    number_of_butches = 200
    testSetSize = 200
    iterations = 1000
    lr = 0.1
    np.random.seed(0)
    hiddenSize = 25
    hiddenLayerAmount = 1

    PeaksData = loadmat('PeaksData.mat')
    trainSet = np.array(PeaksData['Yt'])
    trainSetLabels = np.array(PeaksData['Ct'])

    validationSet = np.array(PeaksData['Yv'])
    validationSetLabels = np.array(PeaksData['Cv'])

    #  shuffle validation set
    idx = np.random.permutation(len(validationSetLabels[0]))
    testSetX = validationSet[:, idx][:, :testSetSize]
    testSetY = validationSetLabels[:, idx][:, :testSetSize]

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

    # gradient test
    # WeightsGradientTest(theta,bias,trainSetX_batches[0],trainSetY_batches[0])
    network_hidden_dims = [2,5, 6, 8 , 10, 5]

    nn_model = Functions.NN(network_hidden_dims, input_size, output_size)

    # JacobianTestHidden(nn_model, trainSetX_batches[0][:, 0].reshape(2, 1), trainSetY_batches[0][:, 0].reshape(5, 1))

    for i in range(0, iterations):
        for batchX, batchY in zip(trainSetX_batches, trainSetY_batches):
            loss, linearLayerArr, nonlinearLayerArr = nn_model.nnForward(batchX, batchY)
            theta_grads, bias_grads = nn_model.backpropagation(linearLayerArr, nonlinearLayerArr, batchY)
            nn_model.step(lr, theta_grads, bias_grads)

        _, linearLayerArr, nonlinearLayerArr = nn_model.nnForward(testSetX, testSetY)
        acc = accuracy(nonlinearLayerArr[-1], testSetY)
        if (i+1)%11==0:
            print(f"iter number:{i}/{iterations}\tloss: {loss}\taccuracy:{acc}%")
