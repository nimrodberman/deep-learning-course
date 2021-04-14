import numpy as np
import matplotlib.pyplot as plt
from work1.task1.Functions import *

epsilonArray = [np.power(0.5, i) for i in range(0, 10)]
rangeArr = [i for i in range(0, 10)]


def WeightsGradientTest(theta, b, xs, ys):
    firstOrderArr = []
    secondOrderArr = []

    cost, grad_theta, _, _ = softmaxRegression(theta, xs, b, ys)
    d = np.random.rand(grad_theta.shape[0], grad_theta.shape[1])
    d = d / np.linalg.norm(d)
    d_vec = d.flatten()
    grad = grad_theta.flatten()

    for eps in epsilonArray:
        d_theta = theta.copy()
        d_theta = d_theta + d * eps
        d_cost = softmaxRegression(d_theta, xs, b, ys)[0]
        firstOrderArr.append(abs(d_cost - cost))
        secondOrderArr.append(abs(d_cost - cost - eps * d_vec.T @ grad))

    plt.plot(rangeArr, firstOrderArr, label="first-order")
    plt.plot(rangeArr, secondOrderArr, label="second-order")
    plt.yscale("log")
    plt.xscale("log")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('epsilons')
    plt.ylabel('absolute differance')
    plt.title('wights gradient test:')
    plt.show()


def BiasGradientTest(theta, b, xs, ys):
    firstOrderArr = []
    secondOrderArr = []

    cost, grad_theta, grad_b, _ = softmaxRegression(theta, xs, b, ys)
    d = np.random.rand(grad_b.shape[0])
    d_vec = d / np.linalg.norm(d)
    grad = grad_b

    for eps in epsilonArray:
        d_b = b.copy()
        d_b = d_b + d * eps
        d_cost = softmaxRegression(theta, xs, d_b, ys)[0]
        firstOrderArr.append(abs(d_cost - cost))
        secondOrderArr.append(abs(d_cost - cost - eps * d_vec.T @ grad))

    plt.plot(rangeArr, firstOrderArr, label="first-order")
    plt.plot(rangeArr, secondOrderArr, label="second-order")
    plt.yscale("log")
    plt.xscale("log")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('epsilons')
    plt.ylabel('absolute differance')
    plt.title('bias gradient test:')
    plt.show()



