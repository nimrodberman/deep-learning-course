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
        self.weightsArray = []
        self.biasArray = []
        self.initParameters()

    def initParameters(self):
        # set the input layer
        self.theta_in = np.random.rand(self.inputSize, self.hiddenSize)
        self.bias_in = np.zeros([1, self.hiddenSize])
        self.weightsArray.append(self.theta_in)
        self.biasArray.append(self.bias_in)

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
        grad = -(1) * target.T - probs
        loss = (-1 / probs.shape[1]) * (np.sum(target.T * np.log(probs)))
        return loss, grad

    def forward(self, X):
        layerOutputs = []
        layerOutputs.append(X)

        X_L = X.copy()
        # hidden layer forward with tanh
        for w, b in zip(self.weightsArray, self.biasArray):
            X_L = np.tanh(np.dot(np.transpose(X_L), w) + b).T
            layerOutputs.append(X_L)

        # softmax output layer
        X_out = np.dot(np.transpose(X_L), self.theta_out) + self.bias_out
        X_prob = softmax(X_out)
        # layerOutputs.append(X_prob.T)

        return X_prob, layerOutputs

    def backprop(self, lr, dl_dy, layerOutputs):
        theta_grads = []
        bias_grads = []
        # softmax output layer gradients
        dl_dhy = layerOutputs[-1] @ dl_dy
        dl_dby = dl_dy
        theta_grads.append(dl_dhy)
        bias_grads.append(dl_dby)
        dl_dh_curr = self.theta_out @ dl_dy.T
        # running over the layers not include the last one
        for layer, t in zip(reversed(layerOutputs[:-1]), reversed(range(len(layerOutputs) - 1))):
            temp = ((1 - layer ** 2) * dl_dh_curr)
            bias_grads.append(temp)
            theta_grads.append(temp @ layerOutputs[t].T)
            if t > 0:
                dl_dh_curr = self.weightsArray[t-1] @ temp

        bias_grads = list(reversed(bias_grads))
        theta_grads = list(reversed(theta_grads))

        for index in range(len(bias_grads)):
            if index <= len(bias_grads) - 2:
                self.weightsArray[index] = self.weightsArray[index] - theta_grads[index].T * lr
                self.biasArray[index] = self.biasArray[index] - bias_grads[index].T * lr
            else:
                self.theta_out = self.theta_out - dl_dhy.T * lr
                self.bias_out = self.bias_out - dl_dby.T * lr
