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
        self.bias_in = np.zeros([1, self.hiddenSize])
        self.weightsArray.append(theta_in)
        self.biasArray.append(self.bias_in)

        # set hidden layers by the layer number parameters
        for i in range(self.hiddenLayerAmount-1):
            theta = np.random.uniform(-limit, limit, size=(self.hiddenSize, self.hiddenSize))
            # theta = np.random.rand(self.hiddenSize, self.hiddenSize)
            bias = np.zeros([1, self.hiddenSize])
            self.weightsArray.append(theta)
            self.biasArray.append(bias)

        # set the output layer
        self.theta_out=np.random.uniform(-limit, limit, size=(self.hiddenSize, self.outputSize))
        # self.theta_out = np.random.rand(self.hiddenSize, self.outputSize)
        self.bias_out = np.zeros([1, self.outputSize])
        # the weightArray doesnt contain the output layer because we use different activation function
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
            # print(f"XL={X_L.shape}    w={w.shape}       b={b.shape} ")
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

        dl_dby = dl_dy
        # softmax output layer gradients, the gradient of the last layer is
        # dl/dy * last_hidden_layer_values
        theta_out_grad = layerOutputs[-1] @ dl_dy
        bias_out_grad = dl_dby

        dl_dh_curr = self.theta_out @ dl_dy.T
        # running over the hidden layers
        for t in reversed(range(self.hiddenLayerAmount)):
            temp = ((1 - layerOutputs[t+1] ** 2) * dl_dh_curr)
            theta_grads.append(-temp @ layerOutputs[t].T)
            bias_grads.append(-temp)
            dl_dh_curr = self.weightsArray[t] @ temp
        # for layer, t in zip(reversed(layerOutputs[:-1]), reversed(range(len(layerOutputs) - 1))):
        #     temp = ((1 - layer ** 2) * dl_dh_curr)
        #     bias_grads.append(temp)
        #     theta_grads.append(temp @ layerOutputs[t-1].T)
        #     if t > 0:
        #         dl_dh_curr = self.weightsArray[t-1] @ temp

        # temp = ((1 - layerOutputs[1] ** 2) * dl_dh_curr)
        # theta_in_grad = temp @ layerOutputs[0].T
        # bias_in_grad = temp
        bias_grads = list(reversed(bias_grads))
        theta_grads = list(reversed(theta_grads))

        # update wights
        # self.theta_in = self.theta_in - theta_in_grad.T*lr
        # self.bias_in = self.bias_in - np.mean(bias_in_grad ,axis=1).T*lr
        for index in range(self.hiddenLayerAmount):
            self.weightsArray[index] = self.weightsArray[index] - theta_grads[index].T * lr
            self.biasArray[index] = self.biasArray[index] - np.mean(bias_grads[index].T ,axis=0)* lr
            # self.weightsArray[index] = np.clip(self.weightsArray[index] - theta_grads[index].T * lr, -1, 1, self.weightsArray[index])
            # self.biasArray[index] = np.clip((self.biasArray[index] - np.mean(bias_grads[index].T, axis=0) * lr), -1, 1, self.biasArray[index])
        self.theta_out = self.theta_out - theta_out_grad * lr
        self.bias_out = self.bias_out - np.mean(bias_out_grad ,axis=0) * lr
        # self.theta_out = np.clip(self.theta_out - theta_out_grad * lr, -1, 1,self.theta_out)
        # self.bias_out = np.clip(self.bias_out - np.mean(bias_out_grad ,axis=0) * lr, -1, 1,self.bias_out)