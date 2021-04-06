import numpy as np
import matplotlib.pyplot as plt
from work1.task1.Functions import *

epsilonArray = [np.power(0.5, i) for i in range(0, 10)]
rangeArr = [i for i in range(0, 10)]



def WeightsGradientTest(theta, b, xs, ys):
    iter_n = 20
    # diff=np.zeros(iter_n)
    # diff_grad=np.zeros(iter_n)

    firstOrderArr = []
    secondOrderArr = []
    d = np.random.rand(xs.shape[0])
    d = d / np.linalg.norm(d)
    cost, grad_theta, _, _ = softmaxRegression(theta, xs, b, ys)
    grad = grad_theta[:, 0]

    for eps in epsilonArray:
        d_theta = theta.copy()
        d_theta[:, 0] += d * eps
        d_cost = softmaxRegression(d_theta, xs, b, ys)[0]
        # diff[i]=abs(d_cost-fw)
        # diff_grad[i]=abs(d_cost-fw-eps*d.T@grad)
        firstOrderArr.append(abs(d_cost - cost))
        secondOrderArr.append(abs(d_cost - cost - eps * d.T @ grad))


    plt.plot(rangeArr,firstOrderArr,label="first-order")
    plt.plot(rangeArr,secondOrderArr,label="second-order")
    plt.yscale("log")
    plt.xscale("log")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('epsilons')
    plt.ylabel('absolute differance')
    plt.title('wights gradient test:')
    # plt.legend('first order', 'second order')
    plt.show()


# def gradientTest1(x,theta, bias, y_mat):
#     # sample a random unit vector
#     d = np.random.rand(x.shape[0])
#     d = d / np.linalg.norm(d)
#
#     firstOrderArr = []
#     secondOrderArr = []
#
#     cost, grad_theta, grad_b, _ = softmaxRegression(theta, x, bias,y_mat)
#
#     for eps in epsilonArray:
#         theta_diff = theta
#         theta_diff = theta_diff + (d * eps).reshape(2, 1)
#         epsilon_cost = softmaxRegression(theta_diff, x, bias, y_mat)[0]
#
#         firstOrderArr.append(abs(epsilon_cost - cost))
#         secondOrderArr.append(abs(epsilon_cost - cost - (eps * (d.T @ grad_theta))))
#
#     plt.plot(firstOrderArr, epsilonArray, secondOrderArr, epsilonArray)
#     print("finished test 1")
#     return 0;


def BiasGradientTest(X, Y, W, b):
    iter = 20
    diff = np.zeros(iter)
    diff_grad = np.zeros(iter)
    epsilons = [0.5 ** i for i in range(iter)]
    n, m = X.shape
    l = len(b.T)
    d = np.random.rand(l)
    d = d / np.linalg.norm(d)
    fw, grad_theta, grad_b, _ = softmaxRegression(theta_L=W, x_L=X, b_L=b, y_mat=Y)
    for i, eps in enumerate(epsilons):
        b_diff = b.copy()
        b_diff += d * eps
        fw_eps = softmaxRegression(W, X, b_diff, Y)[0]
        diff[i] = np.abs(fw_eps - fw)
        diff_grad[i] = np.absolute(fw_eps - fw - eps * d.T @ grad_b)

    plt.semilogy(np.arange((1, iter + 1, 1), diff))
    plt.semilogy(np.arange((1, iter + 1, 1), diff_grad))
    plt.xlabel('epsilons')
    plt.ylabel('diffrences')
    plt.title('bais gradient test result')
    plt.legend('diff without grad', 'diff with grad')
    plt.show()
