import matplotlib.pyplot as plt
from work2.datasets import *
from work2.models import *

# TODO - 3.1.2 train and report about the grid search
# TODO - 3.1.2 print the original vs reconstruction on a graph

batch_size = 20
data_size = 300
time_size = 20
syntheticDataGenerator = SeriesDataset()
data = syntheticDataGenerator.getSyntheticDataInBatches(data_size, batch_size, time_size)
train_data = syntheticDataGenerator.getSyntheticDataInBatches(6000, batch_size, time_size)
validation_data = syntheticDataGenerator.getSyntheticDataInBatches(2000, 2000, time_size)[0]


def synthetic_data_experiment(optimizer_name, lr, hidden_state_size, epochs, gradient_cliping):
    optimizer_name = optimizer_name
    lr = lr
    weight_decay = 0
    optimizer = None
    gradient_clipping = gradient_cliping
    model = VaLstm(inputSize=1, outputSize=1, hiddenStateSize=hidden_state_size)

    # ------ selecting optimizer ----- #
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    resultLoss = []
    resultAcc = []
    loss = 0
    for epoch in range(epochs):
        # iterate over each batch - in our case tensor.
        for batch in train_data:
            # reset the gradient from previous epochs
            optimizer.zero_grad()
            # feedforward mean encoding y to z and then decoding z to y_hat
            _, y_hat = model.forward(batch)
            # calculate loss
            loss = model.loss(y=batch, y_hat=y_hat)
            # calculate gradients
            loss.backward()
            # gradient clipping
            if gradient_clipping:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            # update weights
            optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            _, val_rec = model.forward(validation_data)
            acc = model.accuracy(validation_data, val_rec)
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            print("Accuracy: {:.4f}".format(acc))
            resultLoss.append(loss.item())
            resultAcc.append(acc)
            model.train()

        # save each 100 epochs the model weights
        if epoch % 100 == 0:
            torch.save(model, 'ae_toy_{}_{}_{}_{}_{}_{}.pt'.format(optimizer_name, lr, hidden_state_size, epochs,
                                                                   gradient_cliping, epoch))

    print('ae_toy_acc_{}_{}_{}_{}_{}'.format(optimizer_name, lr, hidden_state_size, epochs,
                                                                   gradient_cliping))
    print(resultAcc)

    print('ae_toy_loss_{}_{}_{}_{}_{}'.format(optimizer_name, lr, hidden_state_size, epochs,
                                                                   gradient_cliping))
    print(resultLoss)
    # np.savetxt('resultAcc_{}_{}_{}_{}_{}.csv'.format(optimizer_name, lr, hidden_state_size, epochs,
    #                                                  gradient_cliping), resultAcc, delimiter=',')
    # np.savetxt('resultLoss_{}_{}_{}_{}_{}.csv'.format(optimizer_name, lr, hidden_state_size, epochs,
    #                                                   gradient_cliping), resultLoss, delimiter=',')


def printSequenceOnGraph(seq, i, lType):
    x = range(len(seq))
    y = seq
    plt.plot(x, y, label='{} signal {}'.format(lType, i))


def printMultipleSequencesOnGraph(sequences, lType, name):
    for i, seq in enumerate(sequences):
        printSequenceOnGraph(seq, i, lType)
    plt.xlabel('time steps')
    plt.ylabel('signal value')
    plt.title('synthetic signal graph')
    plt.legend()
    plt.show()
    plt.savefig('{}.png'.format(name))


def printReconstrctedAndOriginal(original, reconstructed):
    for i, seq in enumerate(original):
        printSequenceOnGraph(seq, i, 'original')
    for i, seq in enumerate(reconstructed):
        printSequenceOnGraph(seq, i, 'reconstructed')
    plt.xlabel('time steps')
    plt.ylabel('signal value')
    plt.title('synthetic signal graph and its reconstruction')
    plt.legend()
    plt.show()
    plt.savefig('{}.png'.format('comparing toy result'))


# task 3.1.1
# syntheticDataGenerator = SeriesDataset()
# signals = syntheticDataGenerator.getSyntheticData(2, 50).numpy()
# printMultipleSequencesOnGraph(signals, 'original', '2_original_samples')

# task 3.1.2 grid search

# ----- grid search training ----- #
test_optimizer_names = ['Adam', 'RMSprop']
test_lr = [0.1, 0.01, 0.001, 0.0001]
hidden_state_size = [16, 32, 64, 128, 256]
test_epochs = [100]
test_gradient_clipping = [True, False]

# for opt in test_optimizer_names:
#     for lr in test_lr:
#         for hidden_size in hidden_state_size:
#             for ep in test_epochs:
#                 for clipping in test_gradient_clipping:
#                     synthetic_data_experiment(opt, lr, hidden_size, ep, clipping)

synthetic_data_experiment('Adam', 0.001, 128, 500, False)


# # # we pick manually the best parameters by the validation set
# # # after picking the hyper parameters
# optimal_opt = 'Adam'
# optimal_lr = 0.001
# optimal_hidden_size = 64
# optimal_epochs = 500
# optimal_clipping = False
# epoch = 200
# model = torch.load('ae_toy_{}_{}_{}_{}_{}_{}.pt'.format(optimal_opt, optimal_lr, optimal_hidden_size, optimal_epochs,
#                                                         optimal_clipping, epoch))

# ----- test best model ----- #
# # test set evaluation on the chosen hyper parameters
# model.eval()
# test_data = syntheticDataGenerator.getSyntheticDataInHotVector(2000, 2000, time_size)[0]
# _, test_y_hat = model.forward(test_data)
# # calculate loss
# loss = model.loss(y=test_data, y_hat=test_y_hat)

# ----- reconstruction best model ----- #
# # reconstruction demonstration on two examples with our best model
# with torch.no_grad():
#     original_data_sample = syntheticDataGenerator.getSyntheticData(2, 15)
#     reconstructed_data = model.forward(syntheticDataGenerator.toHotVec(original_data_sample))[1]
#     original_data_sample = original_data_sample.numpy()
#     preds = torch.argmax(reconstructed_data, dim=2).numpy()
#     printReconstrctedAndOriginal(reconstructed=preds, original=original_data_sample)
