import matplotlib.pyplot as plt
from work2.datasets import *
from work2.models import *

# TODO - 3.1.2 train and report about the grid search
# TODO - 3.1.2 print the original vs reconstruction on a graph

num_of_epochs = 10000
batch_size = 20
# data_size = 10000
# time_size = 1007
time_size = 50
# input_size = 5
input_size = 1
data_gen = sp500Dataset()
train_data, validation_data = data_gen.getDataForModel(batch_size, time_size)
validation_data = validation_data.reshape(len(validation_data), time_size, input_size)

lr=0.0001
print_every = 10  # epochs
# open, high, low, close, volume = 0, 1, 2, 3, 4
attribute_dict = {
    'open': 0
    # "high": 1,
    # "low": 2,
    # "close": 3,
    # "volume": 4
}
hidden_state_size = 128


def snp500_data_experiment(optimizer_name, lr, hidden_state_size, epochs, gradient_cliping):
    optimizer_name = optimizer_name
    lr = lr
    weight_decay = 0
    optimizer = None
    gradient_clipping = gradient_cliping
    model = VaLstm(inputSize=input_size, outputSize=input_size, hiddenStateSize=hidden_state_size, classification=False,
                   sp500Pred=False)
    model = model.float()



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
            batch = batch.reshape(len(batch), time_size, input_size)
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

        if epoch % print_every == 0:
            model.eval()
            _, val_rec = model.forward(validation_data)
            acc = model.accuracy(validation_data, val_rec)
            print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
            print("Accuracy: {:.4f}".format(acc))
            resultLoss.append(loss.item())
            resultAcc.append(acc)
            model.train()
            printReconstrctedAndOriginal(batch[:3], y_hat[:3].detach().numpy())
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


def printSequenceOnGraph(seq, i, lType, feature, linestyle='solid'):
    x = range(len(seq))
    y = seq[:, feature]
    plt.plot(x, y, linestyle=linestyle, label='{} stock {}'.format(lType, i))


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
    # for i, seq in enumerate(original):
    #     printSequenceOnGraph(seq, i, 'original',feature=high)
    # for i, seq in enumerate(reconstructed):
    #     printSequenceOnGraph(seq, i, 'reconstructed',feature=high, linestyle='dashed')

    original = original[0]
    reconstructed = reconstructed[0]
    for i, atr in enumerate(attribute_dict):
        printSequenceOnGraph(original, atr, 'original', feature=i)
    for i, atr in enumerate(attribute_dict):
        printSequenceOnGraph(reconstructed, atr, 'reconstructed', feature=i, linestyle='dashed')
    plt.xlabel('date')
    plt.ylabel('high value')
    plt.title('S&P500 Ground Truth vs. Reconstruction')
    plt.legend()
    plt.show()
    # plt.savefig('{}.png'.format('comparing toy result'))


# task 3.1.1
# syntheticDataGenerator = SeriesDataset()
# signals = syntheticDataGenerator.getSyntheticData(2, 50).numpy()
# printMultipleSequencesOnGraph(signals, 'original', '2_original_samples')

# task 3.1.2 grid search


snp500_data_experiment('Adam', lr, hidden_state_size, num_of_epochs, False)
