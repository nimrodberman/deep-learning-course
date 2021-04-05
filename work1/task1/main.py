import numpy as np
from work1.task1 import Functions
from work1.task1.GradientTest import *
from scipy.io import loadmat

if __name__ == '__main__':
    print("main started")
    sample_size=20
    PeaksData = loadmat('PeaksData.mat')
    trainSet = np.array(PeaksData['Yt'])
    trainSetLabels = np.array(PeaksData['Ct'])

    idx = np.random.permutation(len(trainSetLabels[0]))[:sample_size]
    batch_x=trainSet[:,idx]
    batch_y=trainSetLabels[:,idx]
    output_size=len(batch_y[:,0])
    input_size=batch_x.shape[0]
    theta = np.random.rand(input_size, output_size)
    b_l = np.zeros([1, output_size])


    _, grad_theta, grad_b, _= Functions.softmaxRegression(theta,batch_x,b_l,batch_y)
    gradientTest1(batch_x,grad_theta,Functions.softmax,theta,b_l,batch_y)#send bias grad

    Bias_Grad_Test(batch_x, batch_y, theta, b_l)

    # iterations = 1000
    # learningRate = 1e-5
    # losses = []
    # for i in range(0, iterations):
    #     loss, grad_theta, grad_b = Functions.softmaxRegression(theta, batch_x, b_l, batch_y)
    #     losses.append(loss)
    #     theta = theta - (learningRate * grad_theta)
    #     grad_b = grad_b - (learningRate * grad_b)
    #     if i % 10 == 0:
    #         print(f"iter number:{i}/{iterations}   loss: {loss}")






    cost, grad_theta, grad_b = Functions.softmaxRegression(theta, x, bias, y_mat)
    GradientTest.gradientTest1(x, grad_b, Functions.softmax, theta, grad_theta)
