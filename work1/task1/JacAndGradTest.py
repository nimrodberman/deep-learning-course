import numpy as np

import matplotlib.pyplot as plt
from Functions import *

epsilonArray = [np.power(0.5, i) for i in range(1, 10)]
rangeArr = [i for i in range(0, len(epsilonArray))]
np.random.seed(5)


def WeightsGradientTest(theta, b, xs, ys):
    firstOrderArr = []
    secondOrderArr = []

    costs = []
    dcosts = []

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

        costs.append(cost / 5)
        dcosts.append(d_cost / 5)

    plt.plot(rangeArr, firstOrderArr, label="first-order")
    plt.plot(rangeArr, secondOrderArr, label="second-order")
    plt.yscale("log")

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



def get_result_and_grads(test, X, hot_vec, nn: NN, l):
    if test == 'loss':
        nonlinear_res, probs = nn.softmaxLayer(target=hot_vec, theta_n=nn.thetasArray[l], x_prev_n=X,
                                               b_n=nn.biasArray[l])
        dW, db, _ = nn.softmaxGradient(target=hot_vec, theta_n=nn.thetasArray[l], x_n=probs, x_prev_n=X)
    elif test == 'hidden':
        linear_res, nonlinear_res = nn.layerForward(theta_n=nn.thetasArray[l], x_n_prev=X, b_n=nn.biasArray[l])
        n, batch_size = linear_res.shape
        # for single layer check we use 1 matrix
        dH_next = np.ones((n, batch_size))
        dW, db, _ = nn.hiddenLayerGradient(theta_n=nn.thetasArray[l], x_n=linear_res, x_prev_n=X, x_next_grad=dH_next)

    return nonlinear_res, dW, db


def jacobian_check(test='loss', batch_size=1, L=1):
    print("run tests")
    if test != 'all':
        L = 1
    full_test = test
    # chose layers sizes
    layers_dim = list(np.random.randint(2, 25, L + 1))
    # layers_dim = [2,5]
    input_size = layers_dim[0]
    number_of_labels = output_size = layers_dim[-1]

    # init wights
    nn_model = NN(layers_dim, input_size, output_size)

    # init labels and samples
    # X = np.random.randn(input_size, batch_size)
    Y = np.random.choice(range(number_of_labels), size=batch_size)
    y_hot_vec = np.zeros((number_of_labels, batch_size))
    y_hot_vec[Y, np.arange(batch_size)] = 1

    # init perpetuated parameters
    deltaTheta = [np.random.random(theta.shape) for theta in nn_model.thetasArray]
    deltaBias = [np.random.random(len(bias)) for bias in nn_model.biasArray]

    original_thetas = nn_model.thetasArray.copy()
    original_bias = nn_model.biasArray.copy()

    if full_test == 'all':
        test = 'hidden'


    for l in range(L):
        last_res = last_res_p = 0
        firstOrderArr = []
        secondOrderArr = []
        if (full_test == 'all') and (l == L - 1):
            test = 'loss'
        X = np.random.randn(nn_model.thetasArray[l].shape[1], batch_size)  # init last hidden layer

        result, dW, db = get_result_and_grads(test=test, X=X, hot_vec=y_hot_vec, nn=nn_model, l=l)

        for eps in epsilonArray:
            # replace the weight with perpetuated weight
            nn_model.thetasArray[l] = original_thetas[l] + eps * deltaTheta[l]

            result_pert, _, _ = get_result_and_grads(test=test, X=X, hot_vec=y_hot_vec, nn=nn_model, l=l)

            if test == 'hidden':
                grad_delta = np.sum((eps * deltaTheta[l]) * dW, axis=1, keepdims=True)
            elif test == 'loss':
                grad_delta = ((eps * deltaTheta[l]).reshape(-1, 1).T @ dW.reshape(-1, 1)).item()

            differance = np.linalg.norm(last_res / (result_pert - result))
            last_res = (result_pert - result)
            differance_prt = np.linalg.norm(last_res_p / (result_pert - result - grad_delta))
            last_res_p = (result_pert - result - grad_delta)
            firstOrderArr.append(differance)
            secondOrderArr.append(differance_prt)

        plt.plot(rangeArr, firstOrderArr, label="first-order")
        plt.plot(rangeArr, secondOrderArr, label="second-order")
        plt.yscale("log")

        plt.legend(loc='lower left', borderaxespad=0.)
        plt.xlabel('epsilons')
        plt.ylabel('differance')
        plt.title(f'Gradient Test For W{l}')
        plt.show()

        # same for bias

        nn_model.thetasArray = original_thetas.copy()
        last_res = last_res_p = 0
        firstOrderArr = []
        secondOrderArr = []

        for eps in epsilonArray:
            # replace the weight with perpetuated weight
            nn_model.biasArray[l] = original_bias[l] + eps * deltaBias[l].reshape(len(nn_model.biasArray[l]), 1)

            result_pert, _, _ = get_result_and_grads(test=test, X=X, hot_vec=y_hot_vec, nn=nn_model, l=l)

            if test == 'hidden':
                grad_delta = np.sum((eps * deltaBias[l]) * db, axis=1, keepdims=True)
            elif test == 'loss':
                grad_delta = ((eps * deltaBias[l]).reshape(-1, 1).T @ db.reshape(-1, 1)).item()

            differance = np.linalg.norm(last_res / (result_pert - result))
            last_res = (result_pert - result)
            differance_prt = np.linalg.norm(last_res_p / (result_pert - result - grad_delta))
            last_res_p = (result_pert - result - grad_delta)
            firstOrderArr.append(differance)
            secondOrderArr.append(differance_prt)

        plt.plot(rangeArr, firstOrderArr, label="first-order")
        plt.plot(rangeArr, secondOrderArr, label="second-order")
        plt.yscale("log")

        plt.legend(loc='lower left', borderaxespad=0.)
        plt.xlabel('epsilons')
        plt.ylabel('differance')
        plt.title(f'Gradient Test For b{l}')
        plt.show()
        nn_model.biasArray = original_bias.copy()


if __name__ == "__main__":
    np.random.seed(40)
    jacobian_check(test='hidden', batch_size=10)
    jacobian_check(test='loss',batch_size=1)
    jacobian_check(test='all', L=6, batch_size=1)
