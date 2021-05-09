from work2.datasets import *
from work2.models import *
# TODO - 3.1.1 print a graph? understand what the meaning and do it.
# TODO - 3.1.2

batch_size = 20
data_size = 100000
time_size = 10
optimizer_name = 'Adam'
lr = 0.01
weight_decay = 0
optimizer = None
gradient_clipping = True
syntheticDataGenerator = SeriesDataset()
data = syntheticDataGenerator.getSyntheticDataInHotVector(data_size, batch_size, time_size)
model = VaLstm(inputSize=10, outputSize=10, hiddenStateSize=256)

# ------ selecting optimizer ----- #
if optimizer_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

resultLoss = []
resultAcc = []
# iterate over each batch - in our case tensor.
for i, batch in enumerate(data):
    # reset the gradient from previous epochs
    optimizer.zero_grad()
    # feedforward mean encoding y to z and then decoding z to y_hat
    _, y_hat = model.forward(batch)
    # calculate loss
    loss = model.loss(y=batch, y_hat=y_hat)
    # calculate gradients
    loss.backward()
    # update weights
    optimizer.step()

    if gradient_clipping:
        nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

    if i % 10 == 0:
        print('Epoch: {}/{}.............'.format(i, len(data)), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        print("Accuracy: {:.4f}".format(model.accuracy(batch, y_hat)))