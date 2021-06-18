import torch
from torch import nn
import torchvision.models as models


class TimeLSTM(nn.Module):
    def __init__(self, feature_size, max_length=20):
        super(TimeLSTM, self).__init__()
        self.cnn = models.resnet152(pretrained=True)
        self.cnn_fc = nn.Linear(feature_size, 11)
        self.set_cnn_as_predictor()

        self.lstm = nn.LSTM(feature_size, hidden_size=512, num_layers=1, batch_first=True)
        self.linear_rsd = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.last_state = None
        self.max_length = max_length

    def set_cnn_as_feature_extractor(self):
        self.cnn.fc = torch.nn.Identity()

    def set_cnn_as_predictor(self):
        self.cnn.fc = self.cnn_fc

    def cnn_features(self, images):
        self.set_cnn_as_feature_extractor()
        return self.cnn(images)

    def cnn_predictions(self, images):
        self.set_cnn_as_predictor()
        return self.cnn(images)

    def rnn_prediction(self, features, stateful=False):
        init_state = self.last_state if stateful else None
        lstm_output, last_state = self.lstm(features, init_state)
        # break graph for last state for truncated backpropagation
        self.last_state = (last_state[0].detach(), last_state[1].detach())

        rsd_pred = self.sigmoid(self.linear_rsd(lstm_output))
        return rsd_pred

    def forward(self, images, dummy=None, stateful=False):
        cnn_features = self.cnn_features(images)
        rsd_prediction = self.rnn_prediction(cnn_features.unsqueeze(0), stateful)
        rsd_prediction *= self.max_length  # denormalize to minutes
        return rsd_prediction, rsd_prediction  # dummy output for compatibility with other models
