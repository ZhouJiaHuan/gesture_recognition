# Description: LSTM model
# Author: ZhouJH
# Data: 2020/4/8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from gesture_lib.model.focal_loss import FocalLoss
from gesture_lib.model.registry import MODELS

loss_dict = {"CELoss": nn.CrossEntropyLoss(),
             "FocalLoss": FocalLoss(4)}


@MODELS.register_module
class LSTM(nn.Module):
    '''
    model structure:

        lstm -> fc1(fc->bn->relu) -> fc2 -> out

        dropout is placed in lstm layer and fc1 layer with ratio 0.5
    '''
    def __init__(self,
                 input_size,
                 cls_num,
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.5,
                 fc_size=512,
                 loss_func='CELoss'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.cls_num = cls_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_layer = nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 dropout=dropout)

        self.fc_layer1 = nn.Sequential(nn.Linear(hidden_size, fc_size),
                                       nn.BatchNorm1d(fc_size),
                                       nn.ReLU(True),
                                       nn.Dropout(0.5))
        self.fc_layer2 = nn.Linear(fc_size, cls_num)

        # self._init_weight()
        assert loss_func in loss_dict.keys()
        self.loss_func = loss_dict[loss_func]
        # self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = FocalLoss(self.cls_num)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                m.all_weights[0][0].data.normal_(0, 0.02)
                m.all_weights[0][1].data.normal_(0, 0.02)
                m.all_weights[1][0].data.normal_(0, 0.02)
                m.all_weights[1][1].data.normal_(0, 0.02)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        out, (h_n, h_c) = self.rnn_layer(x)
        out = out[:, -1, :]
        out = self.fc_layer1(out)
        out = self.fc_layer2(out)
        return out

    def loss(self, out, label):
        return self.loss_func(out, label)
