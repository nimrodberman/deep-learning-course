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
    preds = np.argmax(prob, axis=0)
    res = np.argmax(testSetY.T, axis=1)

    accuracy = sum(preds == res) / (float(len(testSetY.T))) * 100
    return accuracy


class NN:
    def __init__(self, hidden_network_dimensions, input_size, output_size):
        self.output_size = output_size
        self.input_size = input_size
        self.thetasArray = []
        self.biasArray = []

        self.initParameters(hidden_network_dimensions)

    def initParameters(self, hidden_network_dimensions):
        # collect the weights with normal distribution
        np.random.seed(0)
        scale = 1 / max(1., (2 + 2) / 2.)
        limit = math.sqrt(3.0 * scale)

        # set weights
        for i in range(1, len(hidden_network_dimensions)):
            theta = np.random.uniform(-limit, limit,
                                      size=(hidden_network_dimensions[i], hidden_network_dimensions[i - 1]))
            bias = np.zeros([hidden_network_dimensions[i], 1])
            self.thetasArray.append(theta)
            self.biasArray.append(bias)

    """
    Description: this function calculate the gradient of each hidden layer
    @:param - theta_n - (h_n, h_n-1) weight matrix 
    @:param - x_n - (h_n, batch_size) layer matrix
    @:param - x_prev_n - (h_n-1, batch_size) layer matrix
    @:param - the "v" (n_h, m) vector that we learned from class that propagate backward
    @:return - dl_dtehta_n - the gradient of the n' layer matrix
    @:return - dl_db_n - the gradient of the n' layer bias
    @:return - dl_dv_new - (h_n-1,batch_size) passed v vector to the next layer
    """

    def hiddenLayerGradient(self, theta_n, x_n, x_prev_n, x_next_grad):
        batch_size = x_n.shape[1]

        # computes the gradient of the layer
        dl_dx_n = (1 - np.tanh(x_n) ** 2)
        # compute the gradients
        common = (dl_dx_n * x_next_grad)
        dl_dtheta_n = (1 / batch_size) * common @ x_prev_n.T
        dl_db_n = (1 / batch_size) * np.sum(common, axis=1, keepdims=True)
        dl_dv_new = theta_n.T @ common

        return dl_dtheta_n, dl_db_n, dl_dv_new

    """
    Description: this function calculate the gradient of a softmax layer
    @:param - target - (class_number, batch_size) one hot vector labels matrix
    @:param - theta_n - (h_n,h_n-1) weight matrix
    @:param - x_n - (h_n, batch_size) layer matrix
    @:param - x_prev_n - (h_n-1, batch_size) layer matrix
    @:return - dl_db_n - the gradient of the bias (h_n,1)
    @:return - dl_dtheta_n - the gradient of the theta weight (h_n , h_n-1)
    @:return - dl_dx_prev_n - the gradient of the previous layer
    """

    def softmaxGradient(self, target, theta_n, x_n, x_prev_n):
        batch_size = target.shape[1]

        # compute softmax gradient
        dl_dy = (1 / batch_size) * (x_n - target)
        dl_dtheta_n = dl_dy @ x_prev_n.T
        dl_db_n = np.sum(dl_dy, axis=1, keepdims=True)
        dl_dx_prev_n = theta_n.T @ dl_dy

        return dl_dtheta_n, dl_db_n, dl_dx_prev_n

    """
    Description: last layer of the the nn net.
    This layer returns the odds of each class on a n' class vector
    h_n = class_number

    @:param - target - (class_number, batch_size) one hot vector labels matrix
    @:param - theta_n - (h_n, h_n-1) weight matrix
    @:param - x_prev_n - (h_n-1, batch_size) layer matrix
    @:param - b_n - (h_n, 1) bias vector
    @:return - cost - a scalar the manifest the total price the net is paying for each mistake 
    @:return - probs - (class_number, batch_size) matrix of the classes probabilities
    """

    def softmaxLayer(self, target, theta_n, x_prev_n, b_n):
        batch_size = b_n.shape[0]

        # calculate the scalar scores for each class
        scores = theta_n @ x_prev_n + b_n
        # finding eta (maximum)
        scores -= np.max(scores)
        # calculate softmax probabilities distribution
        probs = (np.exp(scores) / np.sum(np.exp(scores), axis=0))
        # calculate the log loss (cross entropy)
        cost = (-1 / batch_size) * (np.sum(np.log(probs) * target))

        return cost, probs

    """
    Description: the forward unit of a layer
    @:param - theta_n - (h_n, h_n-1) weight matrix
    @:param - x_prev_n - (h_n-1, batch_size) layer matrix
    @:param - b_n - (h_n, 1) bias vector 
    @:return - linear_res - (h_n, batch_size) linear transform to new layer
    @:return - nonlinear_res - (h_n, batch_size) non linear transform to new layer 
    with a tanh nonlinear function
    """

    def layerForward(self, theta_n, x_n_prev, b_n):
        linear_res = theta_n @ x_n_prev + b_n
        nonlinear_res = np.tanh(linear_res)

        return linear_res, nonlinear_res

    """
    Description: calculates one forward pass through the all network
    @:param - inputs - (h_0, batch_size)
    @:param - target - (class_number, batch_size) one hot vector labels matrix
    @:param - 
    """

    def nnForward(self, inputs, target):
        linearLayerArr = [inputs.copy()]
        nonlinearLayerArr = [inputs.copy()]

        # propagate forward though each layer
        for i in range(0, len(self.thetasArray)-1):
            linear_res, nonlinear_res = self.layerForward(self.thetasArray[i],
                                                          nonlinearLayerArr[i],
                                                          self.biasArray[i])
            linearLayerArr.append(linear_res)
            nonlinearLayerArr.append(nonlinear_res)

        # calculate the last layer(softmax) result and the cost function
        cost, probs = self.softmaxLayer(target, self.thetasArray[-1],
                                        nonlinearLayerArr[-1], self.biasArray[-1])
        nonlinearLayerArr.append(probs)

        return cost, linearLayerArr, nonlinearLayerArr

    """
    Description: calculate the gradient of each parameter

    """

    def backpropagation(self, linearArray, nonlinearArray, target):
        theta_grads = []
        bias_grads = []
        layer_number = len(nonlinearArray) -1
        # last layer gradient computation
        last_layer = nonlinearArray[layer_number]
        prev_last_layer = nonlinearArray[layer_number-1]
        dl_dtheta_n, dl_db_n, dl_dx_prev_n = self.softmaxGradient(target, self.thetasArray[layer_number-1], last_layer, prev_last_layer)
        theta_grads.append(dl_dtheta_n)
        bias_grads.append(dl_db_n)

        # calculate each layer parameters gradient
        for i in range(layer_number - 1, 0, -1):
            dl_dh_next = dl_dx_prev_n.copy()
            x_n = linearArray[i]
            prev_x_n = nonlinearArray[i - 1]
            w = self.thetasArray[i - 1]

            dl_dtheta_n, dl_db_n, dl_dx_prev_n  = self.hiddenLayerGradient(w, x_n, prev_x_n, dl_dh_next)

            theta_grads.append(dl_dtheta_n)
            bias_grads.append(dl_db_n)

        return list(reversed(theta_grads)), list(reversed(bias_grads))

    def step(self, lr, theta_grads, bias_grads):
        for i in range(len(self.thetasArray)):
            self.thetasArray[i] = self.thetasArray[i] - lr * theta_grads[i]
            self.biasArray[i] = self.biasArray[i] - lr * bias_grads[i]
