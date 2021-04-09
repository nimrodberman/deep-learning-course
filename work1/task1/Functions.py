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
    def __init__(self, inputSize, outputSize, hiddenSize, layerNumber):
        # model parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.layerNumber = layerNumber
        # model architecture
        self.initParameters(inputSize, outputSize, hiddenSize, layerNumber)
        self.weightsArray = []
        self.biasArray = []

    def initParameters(self):
        # set the input layer
        self.theta_in = np.random.rand(self.inputSize, self.hiddenSize)
        self.bias_in = np.zeros([1, self.hiddenSize])

        # set hidden layers by the layer number parameters
        for i in range(self.layerNumber - 2):
            theta = np.random.rand(self.hiddenSize, self.hiddenSize)
            bias = np.zeros([1, self.hiddenSize])
            self.weightsArray.append(theta)
            self.biasArray.append(bias)

        # set the output layer
        self.theta_out = np.random.rand(self.hiddenSize, self.outputSize)
        self.bias_out = np.zeros([1, self.outputSize])

    def lossFunction(self, probs, target):
        # return log loss average
        return (-1 / probs.shape[1]) * (np.sum(target.T * np.log(probs)))

    def forward(self, X):
        layerOutputs = []

        # input layer forward with tanh
        X_1 = np.tanh(np.dot(np.transpose(X), self.theta_in) + self.bias_in)
        layerOutputs.append(X_1)

        X_L = X_1.copy()
        # hidden layer forward with tanh
        for w, b in zip(self.weightsArray, self.biasArray):
            X_L = np.tanh(np.dot(np.transpose(X_L), w) + b)
            layerOutputs.append(X_L)

        # softmax output layer
        X_out = np.dot(np.transpose(X_L), self.theta_out) + self.bias_out
        X_prob = softmax(X_out)
        layerOutputs.append(X_prob)

        return X_prob

    def backprop(self, probs, target, x_l, lr):
        gradArray = []
        # softmax output layer gradients
        out_grad_theta = (-1 / x_l.shape[1]) * (x_l @ (target.T - probs))
        out_grad_b = -(1 / x_l.shape[1]) * np.sum(target.T - probs, axis=0).T
        gradArray.append(out_grad_theta)
        gradArray.append(out_grad_b)

        # other tanh layers gradients


        return 0;

    def accuracy(self):
        return 0;
