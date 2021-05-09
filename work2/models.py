import torch.nn as nn
import torch

class VaLstm(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenStateSize):
        super(VaLstm, self).__init__()
        # model parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenStateSize = hiddenStateSize
        # net parameters
        # encoder
        self.encoder = torch.nn.LSTM(input_size=inputSize, hidden_size=hiddenStateSize, batch_first=True)
        # decoders
        self.lstmDecoder = torch.nn.LSTM(input_size=hiddenStateSize, hidden_size=hiddenStateSize,  batch_first=True)
        self.linearDecoder = torch.nn.Linear(hiddenStateSize, outputSize)
        # loss function
        self.loss_function = nn.MSELoss()
        # self.initParameters()

    # def initParameters(self):
    #     nn.init.kaiming_normal_(self.wOutput.weight.data, nonlinearity="relu")
    #     nn.init.constant_(self.wOutput.bias.data, 0)

    def forward(self, inputs):
        # encode the inputs using lstm and get h_n
        encoded_inputs, _ = self.encoder(inputs)
        # get h_T (last hidden state) and make z = h_T
        z = encoded_inputs[: , -1]
        # expand z to be T times
        expand_z = z.repeat(1, inputs.shape[1]).view(encoded_inputs.shape)
        # decode to get hidden states
        decoded_hidden_state, _ = self.lstmDecoder(expand_z)
        # reconstruct to the pixel space
        reconstructed_y = self.linearDecoder(decoded_hidden_state)

        return encoded_inputs, reconstructed_y

    def loss(self, y, y_hat):
        return self.loss_function(y, y_hat)

    def accuracy(self, y, y_hat):
        batch_number = y.shape[0]
        time_steps = y.shape[1]
        ground_truth = torch.argmax(y, dim=2)
        reconstruction = torch.argmax(y_hat, dim=2)
        total_hits = 0
        for g, r in zip(ground_truth, reconstruction):
            total_hits += sum(g == r)

        return (total_hits.item() / float(batch_number * time_steps)) * 100


