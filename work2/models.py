#@title model { form-width: "60px" }
import torch.nn as nn
import torch


class VaLstm(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenStateSize, classification=False, labelSize=10, sp500Pred=False):
        super(VaLstm, self).__init__()
        # model parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenStateSize = hiddenStateSize
        # net parameters
        # encoder
        self.encoder = torch.nn.LSTM(input_size=inputSize, hidden_size=hiddenStateSize, batch_first=True)
        # decoders
        self.lstmDecoder = torch.nn.LSTM(input_size=hiddenStateSize, hidden_size=hiddenStateSize, batch_first=True)
        self.linearDecoder = torch.nn.Linear(hiddenStateSize, outputSize)
        # loss function
        self.loss_function = nn.MSELoss()
        self.classification = classification
        self.sp500Pred = sp500Pred

        if self.classification:
            self.classificationLayer = torch.nn.Linear(hiddenStateSize, labelSize)
            self.crossEntropy = nn.CrossEntropyLoss()



    def forward(self, inputs):
        # encode the inputs using lstm and get h_n
        encoded_inputs, _ = self.encoder(inputs)
        # get h_T (last hidden state) and make z = h_T
        z = encoded_inputs[:, -1]
        # expand z to be T times
        expand_z = z.repeat(1, inputs.shape[1]).view(encoded_inputs.shape)
        # decode to get hidden states

        decoded_hidden_state, _ = self.lstmDecoder(expand_z)
        # --- task 3.2 additional experiment change ----
        # decoded_hidden_state, _ = self.lstmDecoder(encoded_inputs)
        # --- task 3.2 additional experiment change ----

        # reconstruct to the pixel space
        reconstructed_y = self.linearDecoder(decoded_hidden_state)

        if self.classification:
            # take the last hidden layer from the decoder
            last_hidden_layer = decoded_hidden_state[:, -1]
            y_tilda = self.classificationLayer(last_hidden_layer)
            return encoded_inputs, reconstructed_y, y_tilda

        else:
            return encoded_inputs, reconstructed_y

    def loss(self, y, y_hat, target=None, y_tilda=None):
        if self.sp500Pred and self.classification:
            return (self.loss_function(y, y_hat) + self.loss_function(target, y_tilda)) / 2
        elif self.classification:
            return (self.loss_function(y, y_hat) + self.crossEntropy(target, y_tilda)) / 2
        else:
            return self.loss_function(y, y_hat)

    def pred_loss(self,target, y_tilda):
        return self.loss_function(target, y_tilda) / 2

    def accuracy(self, y, y_hat):
        batch_number = y.shape[0]
        time_steps = y.shape[1]
        ground_truth = torch.argmax(y, dim=2)
        reconstruction = torch.argmax(y_hat, dim=2)
        total_hits = 0
        for g, r in zip(ground_truth, reconstruction):
            total_hits += sum(g == r)
        return (total_hits.item() / float(batch_number * time_steps)) * 100

    def classification_accuracy(self, target, y_tilda):
        preds = torch.argmax(y_tilda, dim=1)
        res = target
        accuracy = sum(preds == res) / (float(len(target))) * 100
        return accuracy