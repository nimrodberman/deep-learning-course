import numpy as np


def softmaxRegression(theta_L, x_L, b_L, y_mat):
    # finding the maximum
    eta = findEta(theta_L, x_L, b_L)
    # active softmax
    probs = softmax(theta_L, x_L, eta, b_L)
    m = x_L.shape[0]

    cost = (-1/m) * (np.sum(y_mat.t() @ np.log(probs)))
    grad_theta = -(1/m) * (x_L @ (y_mat - probs))
    grad_b = -(y_mat - probs)

    return cost, grad_theta, grad_b


def findEta(theta_L, x_L, b_L):
    calculatedVector = (x_L @ theta_L) + b_L
    return np.max(calculatedVector)


def softmax(theta_L, x_L, eta, b_L):
    expVec = np.exp((x_L @ theta_L + b_L) - eta)
    return expVec / np.sum(expVec)

