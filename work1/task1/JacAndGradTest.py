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


def JacobianTestHidden(nn: NN, xs, ys):
    input_size, batch_size = xs.shape
    u = np.random.rand(nn.hiddenSize, nn.hiddenSize)
    u = u / np.linalg.norm(u)
    probs, layers = nn.forward(xs)
    _ , dl_dy = nn.lossFunction(probs,ys,xs)

    # chose the first layer in the nn
    layer = layers[2]
    theta = nn.weightsArray[1]

    d = np.random.rand(nn.hiddenSize , nn.hiddenSize)
    d = d / np.linalg.norm(d)
    d_flat=d.flatten()

    grads_t = nn.backprop(0, dl_dy, layers, xs)[0]
    grad = grads_t[1]
    grad = grad.flatten()
    gx = np.dot(layer.T, u)

    firstOrderArr=[]
    secondOrderArr=[]

    for eps in epsilonArray:
        d_theta = theta.copy()
        d_theta = d_theta + d * eps
        nn.weightsArray[1]=d_theta

        probs , layers_pertubated = nn.forward(xs)
        gx_eps = np.dot(layers_pertubated[2].T,u)

        firstOrderArr.append(abs(gx_eps - gx))
        secondOrderArr.append(abs(gx_eps - gx - eps * d_flat.T @ grad))

    x=5
