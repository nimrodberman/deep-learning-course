import numpy as np
import matplotlib.pyplot as plt
from work1.task1.Functions import *

epsilonArray = [np.power(0.5, i) for i in range(0, 10)]

def GradientTest(X,Y,W,b):
  iter_n=20
  diff=np.zeros(iter_n)
  diff_grad=np.zeros(iter_n)
  epsilonArray = [np.power(0.5, i) for i in range(0, 10)]
  n,m=X.shape
  d = np.random.rand(X.shape[0])
  d = d / np.linalg.norm(d)
  fw,grads,_ ,_= softmaxRegression(W,X,b,Y)
  grad=grads[:,0]

  for i,eps in enumerate(epsilonArray):
    W_diff=W.copy()
    W_diff[:,0]+=d*eps
    fw_eps=softmaxRegression(W_diff,X,b,Y)[0]
    diff[i]=abs(fw_eps-fw)
    diff_grad[i]=abs(fw_eps-fw-eps*d.T@grad)


def gradientTest1(x,theta, bias, y_mat):
    # sample a random unit vector
    d = np.random.rand(x.shape[0])
    d = d / np.linalg.norm(d)

    firstOrderArr = []
    secondOrderArr = []

    cost, grad_theta, grad_b, _ = softmaxRegression(theta, x, bias,y_mat)

    for eps in epsilonArray:
        theta_diff = theta
        theta_diff = theta_diff + (d * eps).reshape(2, 1)
        epsilon_cost = softmaxRegression(theta_diff, x, bias, y_mat)[0]

        firstOrderArr.append(abs(epsilon_cost - cost))
        secondOrderArr.append(abs(epsilon_cost - cost - (eps * (d.T @ grad_theta))))

    plt.plot(firstOrderArr, epsilonArray, secondOrderArr, epsilonArray)
    print("finished test 1")
    return 0;


def Bias_Grad_Test(X, Y, W, b):
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
