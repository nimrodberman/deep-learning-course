import numpy as np
from networkx.algorithms.tests.test_communicability import scipy


def softmaxRegression(theta_L, x_L, b_L, y_mat):
    # active softmax
    scores = np.dot(np.transpose(x_L), theta_L) + b_L
    probs = softmax(scores)
    m = x_L.shape[1]

    cost = (-1 / m) * (np.sum(y_mat.T * np.log(probs)))
    grad_theta = (-1 / m) * (x_L @ (y_mat.T - probs))
    grad_b = -(1 / m) * np.sum(y_mat.T - probs, axis=0).T

    return cost, grad_theta, grad_b, probs


def softmax(outputLayer):
    # finding the maximum
    outputLayer -= np.max(outputLayer)
    # calculate softmax
    result = (np.exp(outputLayer).T / np.sum(np.exp(outputLayer), axis=1)).T
    return result


def accuracy(prob, testSetY):
    preds = np.argmax(prob, axis=1)
    res = np.argmax(testSetY.T, axis=1)

    accuracy = sum(preds == res) / (float(len(testSetY.T))) * 100
    return accuracy


class NN:
    def __init__(self, inputSize, outputSize, layerNumber):
        # model parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.layerNumber = layerNumber
        # model architecture
        self.initParameters(inputSize,outputSize layerNumber)
        # self.wOutput = nn.Linear(hiddenStateSize,outputSize, bias=True)
        # self.rnn = LvfModel(inputSize, hiddenStateSize)
        # self.lossFunction = nn.CrossEntropyLoss()

    def initParameters(self ,inputSize, outputSize, layerNumber):
        retrun 0;

    def lossFunction(self):
        return 0;

    def forward(self):
        return 0;

    def backprop(self):
        return 0;

    def accuracy(self):
        return 0;