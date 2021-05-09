import torch
from work2.datasets import *
from work2.models import *

# TODO - 3.1.1 print a graph of the data set.
# TODO - 3.1.2 print the original vs reconstruction on a graph
# TODO - 3.1.2 report about the grid search
# TODO split the data into train test and validation

# ----- grid search ----- #
test_optimizer_names = ['Adam', 'RMSprop']
test_lr = [0.1, 0.01, 0.001]
hidden_state_size = [64, 128, 256]
test_epochs = [1000]
test_gradient_clipping = [True, False]


def synthetic_data_experiment(optimizer_name, lr, hidden_state_size, epochs, gradient_cliping):
    batch_size = 20
    data_size = 1000
    time_size = 50
    optimizer_name = optimizer_name
    lr = lr
    weight_decay = 0
    optimizer = None
    gradient_clipping = gradient_cliping
    syntheticDataGenerator = SeriesDataset()
    data = syntheticDataGenerator.getSyntheticDataInHotVector(data_size, batch_size, time_size)
    model = VaLstm(inputSize=10, outputSize=10, hiddenStateSize=hidden_state_size)

    # ------ selecting optimizer ----- #
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    resultLoss = []
    resultAcc = []
    for epoch in range(epochs):
        # iterate over each batch - in our case tensor.
        for batch in data:
            # reset the gradient from previous epochs
            optimizer.zero_grad()
            # feedforward mean encoding y to z and then decoding z to y_hat
            _, y_hat = model.forward(batch)
            # calculate loss
            loss = model.loss(y=batch, y_hat=y_hat)

            if gradient_clipping:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

            # calculate gradients
            loss.backward()
            # update weights
            optimizer.step()

        if epoch % 10 == 0:
            acc = model.accuracy(batch, y_hat)
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            print("Accuracy: {:.4f}".format(acc))
            resultLoss.append(loss.item())
            resultAcc.append(acc)

    print_and_save_result(resultLoss, resultAcc)


# TODO - save a picture of the
def print_and_save_result(acc, loss):
    return 0


synthetic_data_experiment(test_optimizer_name, test_lr, hidden_state_size, test_epochs, test_gradient_clipping)
