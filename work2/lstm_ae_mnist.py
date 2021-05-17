from work2.datasets import *
from work2.models import *
import torch
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np

# # ----- grid search ----- #
# test_optimizer_names = ['Adam', 'RMSprop']
# test_lr = [0.1, 0.01, 0.001]
# hidden_state_size = [64, 128, 256]
# test_epochs = [1000]
# test_gradient_clipping = [True, False]
test_optimizer_names = 'Adam'
test_lr = 0.001
hidden_state_size = 64
test_epochs = 1000
data_size = 4000
print_every = 1  # epoches
test_size = 1000
test_gradient_clipping = True


def serial_mnist_experiment(optimizer_name, lr, hidden_state_size, epochs, gradient_cliping, clasify=True):
    # constants:
    time_size = 28
    input_size = 28
    number_of_labels = 10

    # params:
    batch_size = 20
    optimizer_name = optimizer_name
    lr = lr
    weight_decay = 0
    optimizer = None
    gradient_clipping = gradient_cliping

    model = VaLstm(inputSize=input_size, outputSize=input_size, hiddenStateSize=hidden_state_size,
                   classification=clasify, labelSize=number_of_labels)
    train_loader, test_loader = getMnistDataLoader(batch_size=batch_size, test_size=test_size)

    # chose test sumple
    test_examples = enumerate(test_loader)
    _, (test_images, test_target) = next(test_examples)

    test_images = test_images.reshape(test_size, 28, 28)

    # ------ selecting optimizer ----- #
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    resultLoss = []
    resultAcc = []

    # run model on test sumple
    _, reconstructed_y_test, y_tilda_test = model.forward(test_images)
    # claculate accuracy on test sumple
    acc = model.classification_accuracy(test_target, y_tilda_test)
    print("Accuracy: {:.4f}".format(acc))
    resultAcc.append(acc)

    for epoch in range(epochs):
        # iterate over each batch - in our case tensor.
        for batch, target in islice(train_loader, data_size):
            batch = batch.reshape(batch_size, 28, 28)
            # reset the gradient from previous epochs
            optimizer.zero_grad()

            if clasify:
                # feedforward mean encoding y to z and then decoding z to y_hat
                _, reconstructed_y, y_tilda = model.forward(batch)
                # calculate loss
                loss = model.loss(batch, reconstructed_y, target, y_tilda)

            else:
                _, reconstructed_y = model.forward(batch)
                # calculate loss
                loss = model.loss(batch, reconstructed_y)

            if gradient_clipping:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

            # calculate gradients
            loss.backward()
            # update weights
            optimizer.step()

        print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

        resultLoss.append(loss.item())
        if epoch % print_every == 0:
            # plot reconstruction
            # show_image_vs_reconstruction(batch,reconstructed_y,batch_size)

            # run model on test sumple
            _, reconstructed_y_test, y_tilda_test = model.forward(test_images)
            # claculate accuracy on test sumple
            acc = model.classification_accuracy(test_target, y_tilda_test)
            print("Accuracy: {:.4f}".format(acc))
            resultAcc.append(acc)

            # plot loss and accuracy
            xbar1 = np.arange(epoch + 1)
            plt.plot(xbar1, resultLoss, label='Loss')
            plt.title("Loss vs. Epoches")
            plt.show()
            xbar2 = np.arange(len(resultAcc))
            plt.plot(xbar2, resultAcc, label='Accuracy')
            plt.title("Accuracy vs. Epoches")
            plt.show()

    print_and_save_result(resultLoss, resultAcc)


# TODO - save a picture of the
def print_and_save_result(acc, loss):
    return 0


def show_image_vs_reconstruction(batch, rec_batch, batch_size):
    images = batch.reshape(batch_size, 1, 28, 28)
    rec_image = rec_batch.reshape(batch_size, 1, 28, 28)
    show_batch(images, "Real Data")
    show_batch(rec_image, "Reconstructed Data")
    return


def show_batch(batch, title):
    im = torchvision.utils.make_grid(batch)
    plt.title(title)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()


serial_mnist_experiment('Adam', test_lr, hidden_state_size, test_epochs, test_gradient_clipping, clasify=True)
