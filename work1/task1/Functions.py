import numpy as np
# from networkx.algorithms.tests.test_communicability import scipy
import math


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
    def __init__(self, inputSize, outputSize, hiddenSize, hiddenLayerAmount):
        # model parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize
        self.hiddenLayerAmount = hiddenLayerAmount
        # model architecture
        # weight array will contain the input layer and the hidden layers
        self.weightsArray = []
        self.biasArray = []
        self.initParameters()

    def initParameters(self):
        # collect the weights with normal distribution
        np.random.seed(0)
        scale = 1 / max(1., (2 + 2) / 2.)
        limit = math.sqrt(3.0 * scale)
        # set the input layer
        theta_in = np.random.uniform(-limit, limit, size=(self.inputSize, self.hiddenSize))
        # self.theta_in = np.random.randn(self.inputSize, self.hiddenSize)
        bias_in = np.zeros([1, self.hiddenSize])

        # set hidden layers by the layer number parameters
        for i in range(self.hiddenLayerAmount):
            # if there are 1 hidden layers we will want to include only this matrix
            if i == 0:
                self.weightsArray.append(theta_in)
                self.biasArray.append(bias_in)
            else:
                theta = np.random.uniform(-limit, limit, size=(self.hiddenSize, self.hiddenSize))
                bias = np.zeros([1, self.hiddenSize])
                self.weightsArray.append(theta)
                self.biasArray.append(bias)

        # if hidden layer number is 0 we will want to change the output matrix to the input size
        # the weightArray doesn't contain the output layer because we use different activation function
        if self.hiddenLayerAmount == 0:
            self.hiddenSize = self.inputSize
        # set the output layer
        self.theta_out = np.random.uniform(-limit, limit, size=(self.hiddenSize, self.outputSize))
        self.bias_out = np.zeros([1, self.outputSize])

    def lossFunction(self, probs, target, x_L):

        m = x_L.shape[1]

        # dl_dy as y = h_n @ theta_out + b_n
        grad = target.T - probs
        # return log loss average
        loss = (-1 / probs.shape[0]) * (np.sum(target.T * np.log(probs)))
        return loss, grad

    def forward(self, X):
        layerOutputs = [X.copy()]

        X_L = X.copy()
        # hidden layer forward with tanh
        for w, b in zip(self.weightsArray, self.biasArray):
            X_L = np.tanh(np.dot(np.transpose(X_L), w) + b).T
            layerOutputs.append(X_L)

        # softmax output layer
        X_out = np.dot(np.transpose(X_L), self.theta_out) + self.bias_out
        X_prob = softmax(X_out)

        return X_prob, layerOutputs

    def backprop(self, lr, dl_dy, layerOutputs, x_l):
        theta_grads = []
        bias_grads = []
        batch_size = x_l.shape[1]

        # softmax output layer gradients, the gradient of the last layer is
        # dl/dy * last_hidden_layer_values
        theta_out_grad = (-1 / batch_size) * layerOutputs[-1] @ dl_dy
        bias_out_grad = (-1 / batch_size) * np.sum(dl_dy, axis=0)
        dl_dh_curr = self.theta_out @ dl_dy.T

        # running over the hidden layers and calculate their weights and biases gradient
        for t in reversed(range(self.hiddenLayerAmount)):
            temp = ((1 - layerOutputs[t + 1] ** 2) * dl_dh_curr)
            theta_grads.append((-1 / batch_size) * (temp @ layerOutputs[t].T))
            bias_grads.append((-1 / batch_size) * (np.sum(temp, axis=1)))
            dl_dh_curr = self.weightsArray[t] @ temp

        # reversing the lists for correct order
        bias_grads = list(reversed(bias_grads))
        theta_grads = list(reversed(theta_grads))

        # update wights
        for index in range(self.hiddenLayerAmount):
            self.weightsArray[index] = self.weightsArray[index] - theta_grads[index].T * lr
            self.biasArray[index] = self.biasArray[index] - bias_grads[index].T * lr

        self.theta_out = self.theta_out - theta_out_grad * lr
        self.bias_out = self.bias_out - bias_out_grad * lr
