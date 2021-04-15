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
    n, m = xs.shape

    probs, output_layers = nn.forward(xs)
    _, dl_dy = nn.lossFunction(probs, ys, xs)

    output_layer = output_layers[1]
    original_layer_weight = nn.weightsArray[0].copy()

    out_dimensions = output_layer.shape[0]

    U = np.random.rand(out_dimensions, m)
    U = U / np.linalg.norm(U)

    D = np.random.rand(*original_layer_weight.shape)
    D = D / np.linalg.norm(D)

    g_x = np.dot(output_layer.T, U)

    # probs, output_layers = nn.forward(xs)
    # _, dl_dy = nn.lossFunction(probs, ys, xs)
    output_layers[1] = U

    grads = nn.backprop(0, dl_dy, output_layers, xs)[0]
    W_Jac = grads[0]

    # U = normalize(np.random.rand(out_dimensions, m))
    # original_W = layer.W.copy()
    #
    # ###################
    iter_num = 20
    diff = np.zeros(iter_num)
    diff_grad = np.zeros(iter_num)
    epsilons = [0.5 ** i for i in range(iter_num)]
    # d = normalize(np.random.rand(*layer.W.shape))
    # fw = np.dot(layer.forward(X).T, U).item()
    # _, JacTu_W, _ = layer.backward(U)

    for i, epsilon in enumerate(epsilons):
        W_diff = original_layer_weight.copy()
        W_diff += D * epsilon
        nn.weightsArray[0] = W_diff
        probs_eps, output_layers_eps = nn.forward(xs)
        output_layer_eps = output_layers_eps[1]
        g_x_eps = np.dot(output_layer_eps.T, U)
        # fw_epsilon = np.dot(layer.forward(X).T, U).item()
        diff[i] = abs(g_x_eps - g_x)
        d_flat = D.reshape(-1, 1)
        JacTu_W_flat = W_Jac.reshape(-1, 1)
        diff_grad[i] = abs(g_x_eps - g_x - epsilon * d_flat.T @ JacTu_W_flat)

    plt.semilogy(np.arange(1, iter_num + 1, 1), diff)
    plt.semilogy(np.arange(1, iter_num + 1, 1), diff_grad)
    plt.xlabel('epsilons')
    plt.ylabel('difference')
    plt.title('weights Grad Test Results')
    plt.legend(("diff without grad", "diff with grad"))
    plt.show()

def JacobianTestHidden2(nn: NN, xs, ys):
    input_size, batch_size = xs.shape
    u = np.random.rand(nn.hiddenSize, batch_size)
    u = u / np.linalg.norm(u)

    # calculate feedforward
    probs, forward_layers = nn.forward(xs)
    _, dl_dy = nn.lossFunction(probs, ys, xs)

    # chose the first hidden layer in the nn
    layer = forward_layers[2]
    theta = nn.weightsArray[1]

    d = np.random.rand(nn.hiddenSize, nn.hiddenSize)
    d = d / np.linalg.norm(d)
    d_flat = d.flatten()

    gx = np.dot(layer.T.flatten(), u.flatten()).item()

    grads_t = nn.backprop(0, dl_dy, forward_layers, xs)[0]
    grad = grads_t[2]
    grad = grad.flatten()

    firstOrderArr = []
    secondOrderArr = []

    for eps in epsilonArray:
        d_theta = theta.copy()
        d_theta = d_theta + d * eps
        nn.weightsArray[1] = d_theta

        probs, layers_pertubated = nn.forward(xs)
        gx_eps = np.sum(layers_pertubated[2] * u)

        firstOrderArr.append(abs(gx_eps - gx))
        secondOrderArr.append(abs(gx_eps - gx - eps * d_flat.T @ grad))

    x = 5
