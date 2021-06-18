import torch
from torch import nn
from typing import TypeVar
from torchvision.models import densenet169
import numpy as np
from models.catRSDNet import CatRSDNet, Rnn_Model

T = TypeVar('T', bound='Module')


class CatRSDNet_NL(CatRSDNet):
    """
    CatRSDNet no labels

    Parameters:
    -----------
    max_len: int, optional
      maximum length of videos to process in minutes, default 20 min

    """
    def __init__(self, max_len=20):
        super(CatRSDNet_NL, self).__init__()

        feature_size = 1664
        self.max_len = max_len
        self.cnn = self.initCNN(feature_size)
        self.rnn = Rnn_Model(input_size=feature_size, output_size=1)

    def train(self: T, mode: bool = True) -> T:
        super(CatRSDNet_NL, self).train(mode)

    def initCNN(self, feature_size, n_classes_1=None, n_classes_2=None):
        cnn = densenet169(pretrained=True)
        tmp_conv_weights = cnn.features.conv0.weight.data.clone()
        cnn.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # copy RGB weights pre-trained on ImageNet
        cnn.features.conv0.weight.data[:, :3, :, :] = tmp_conv_weights.clone()
        # compute average weights over RGB channels and use that to initialize the optical flow channels
        # technique from Temporal Segment Network, L. Wang 2016
        mean_weights = torch.mean(tmp_conv_weights[:, :3, :, :], dim=1)
        cnn.features.conv0.weight.data[:, 3, :, :] = mean_weights

        # add classification
        cnn.classifier = torch.nn.Linear(feature_size, 1)
        return cnn

    def forwardRNN(self, X, stateful=False, skip_features=False):
        """

        Parameters
        ----------
         X: torch.tensor, shape(batch_size, 4, 224, 224)
            input image tensor with elapsed time in 4th channel
         stateful: bool, optional
            true -> store last RNN state, default false
         skip_features: bool, optional
            true -> X is input features and by-pass the CNN, default false

         Returns
         -------
         torch.tensor, shape(batch_size,)
            remaining surgical time prediction in minutes

        """
        if skip_features:
            features = X
        else:
            self.set_cnn_as_feature_extractor()
            features = self.cnn(X)
            features = features.unsqueeze(0) # go from [batch size, C] to [batch size, sequence length, C]
        rsd_predictions = self.rnn(features, stateful=stateful)
        return rsd_predictions

    def forward(self, X, elapsed_time=None, stateful=False):
        """

        Parameters
        ----------
         X: torch.tensor, shape(batch_size, k, 224, 224)
            input image tensor, k=3 if elapsed_time provided otherwise k=4 with time_channel
        elapsed_time: float, optional
            elapsed time from start of surgery in minutes. Optional, if timestamp channel is already appended to X.
         stateful: bool, optional
            true -> store last RNN state, default false

         Returns
         -------
         torch.tensor, shape(batch_size,)
            remaining surgical time prediction in minutes

        """
        self.set_cnn_as_feature_extractor()
        # select if
        if elapsed_time is not None:
            assert X.shape[1] == 3, 'provide images with RGB channels only if elapsed time is set.'
            features = self.forwardCNN(X, elapsed_time.view(-1))
        else:
            assert X.shape[1] == 4, 'provide images with RGB+Timestamp if elapsed time is not set.'
            features = self.cnn(X)
        features = features.unsqueeze(0) # go from [batch size, C] to [batch size, sequence length, C]
        rsd_predictions = self.rnn(features, stateful=stateful)
        return rsd_predictions

    def load_catrsdnet(self, checkpoint):
        dummy_CatRSDNet = CatRSDNet()
        dummy_CatRSDNet.set_cnn_as_feature_extractor()
        dummy_CatRSDNet.load_state_dict(checkpoint)
        self.set_cnn_as_feature_extractor()
        self.cnn = dummy_CatRSDNet.cnn
        rnn_fc = dummy_CatRSDNet.rnn.fc.fc3
        self.rnn = dummy_CatRSDNet.rnn
        self.rnn.fc = rnn_fc
