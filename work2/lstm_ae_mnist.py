from work2.datasets import *
from work2.models import *
import torch
from itertools import islice

# 3.2.1 TODO import normalize mnist data using torch vision
# 3.2.1 TODO organize each image data in roes
# 3.2.1 TODO print some reconstruction examples
# 3.2.2 TODO plot loss vs epoch of the


# ----- grid search ----- #
# test_optimizer_names = ['Adam', 'RMSprop']
# test_lr = [0.1, 0.01, 0.001]
# hidden_state_size = [64, 128, 256]
# test_epochs = [1000]
# test_gradient_clipping = [True, False]
test_optimizer_names = 'Adam'
test_lr = 0.001
hidden_state_size = 256
test_epochs = 1000
data_size = 1000
test_gradient_clipping = True


def serial_mnist_experiment(optimizer_name, lr, hidden_state_size, epochs, gradient_cliping):
    # constants:
    time_size = 28
    input_size = 28
    number_of_labels = 10

    # params:
    batch_size = 5
    optimizer_name = optimizer_name
    lr = lr
    weight_decay = 0
    optimizer = None
    gradient_clipping = gradient_cliping

    model = VaLstm(inputSize=input_size, outputSize=input_size, hiddenStateSize=hidden_state_size, classification=True,
                   labelSize=number_of_labels)
    train_loader, test_loader = getMnistDataLoader(batch_size=batch_size)
    # ------ selecting optimizer ----- #
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    resultLoss = []
    resultAcc = []
    for epoch in range(epochs):
        # iterate over each batch - in our case tensor.
        for batch, target in islice(train_loader, data_size):
            batch = batch.reshape(batch_size, 28, 28)
            # reset the gradient from previous epochs
            optimizer.zero_grad()
            # feedforward mean encoding y to z and then decoding z to y_hat
            _, reconstructed_y, y_tilda = model.forward(batch)
            # turn y_tilda to hot vector
            target_hotvec = torch.nn.functional.one_hot(target, number_of_labels).float()

            # calculate loss
            loss = model.loss(batch, reconstructed_y, target, y_tilda)

            if gradient_clipping:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

            # calculate gradients
            loss.backward()
            # update weights
            optimizer.step()

        if epoch % 1 == 0:
            # acc = model.accuracy(batch, y_hat)
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            # print("Accuracy: {:.4f}".format(acc))
            resultLoss.append(loss.item())
            # resultAcc.append(acc)

    print_and_save_result(resultLoss, resultAcc)


# TODO - save a picture of the
def print_and_save_result(acc, loss):
    return 0


serial_mnist_experiment('Adam', test_lr, hidden_state_size, test_epochs, test_gradient_clipping)