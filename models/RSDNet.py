import torch
from torch import nn
import torchvision.models as models


class RSDNet(nn.Module):
    def __init__(self, feature_size):
        super(RSDNet, self).__init__()
        self.cnn = models.resnet152(pretrained=True)
        self.cnn_fc = nn.Linear(feature_size, 1)
        self.set_cnn_as_predictor()

        self.lstm = nn.LSTM(feature_size, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear_progress = nn.Linear(513, 1)
        self.linear_rsd = nn.Linear(513, 1)
        self.sigmoid = nn.Sigmoid()
        self.last_state = None

    def set_cnn_as_feature_extractor(self):
        self.cnn.fc = torch.nn.Identity()

    def set_cnn_as_predictor(self):
        self.cnn.fc = self.cnn_fc

    def cnn_features(self, images):
        self.set_cnn_as_feature_extractor()
        return self.cnn(images)

    def cnn_predictions(self, images):
        self.set_cnn_as_predictor()
        return self.sigmoid(self.cnn(images))

    def rnn_prediction(self, features, elapsed_time, stateful=False):
        init_state = self.last_state if stateful else None
        lstm_output, last_state = self.lstm(self.dropout(features), init_state)
        # break graph for last state for truncated backpropagation
        self.last_state = (last_state[0].detach(), last_state[1].detach())

        lstm_output = self.dropout(lstm_output)
        unsqueezed_elapsed_time = elapsed_time.unsqueeze(2)
        lstm_and_elapsed_time = torch.cat((lstm_output, unsqueezed_elapsed_time), 2)

        progress_pred = self.sigmoid(self.linear_progress(lstm_and_elapsed_time))
        rsd_pred = self.linear_rsd(lstm_and_elapsed_time)
        return progress_pred, rsd_pred

    def forward(self, images, elapsed_time, stateful=False):
        cnn_features = self.cnn_features(images)
        return self.rnn_prediction(cnn_features.unsqueeze(0), elapsed_time, stateful)


