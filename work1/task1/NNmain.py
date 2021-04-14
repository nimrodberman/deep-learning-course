import numpy as np
from work1.task1 import Functions
from work1.task1.JacAndGradTest import *
from scipy.io import loadmat

if __name__ == '__main__':
    print("main started")

    # upload data and set parameters for it
    number_of_butches = 3000
    testSetSize = 200
    iterations = 1000
    lr = 0.0001
    np.random.seed(0)
    hiddenSize = 25
    hiddenLayerAmount = 5

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

    nn_model = Functions.NN(input_size, output_size, hiddenSize=hiddenSize, hiddenLayerAmount=hiddenLayerAmount)

    JacobianTestHidden(nn_model,trainSetX_batches[0],trainSetY_batches[0])

    for i in range(0, iterations):
        for batchX, batchY in zip(trainSetX_batches, trainSetY_batches):
            X_prob, layerOutputs = nn_model.forward(batchX)
            loss, dl_dy = nn_model.lossFunction(probs=X_prob, target=batchY, x_L=batchX)
            nn_model.backprop(lr=lr, dl_dy=dl_dy, layerOutputs=layerOutputs, x_l=batchX)
        if i % 10 == 0:
            X_prob, layerOutputs = nn_model.forward(trainSetX)
            acc = accuracy(X_prob, trainSetY)
            print(f"iter number:{i}/{iterations}   loss: {loss}    accuracy:{acc}%")
