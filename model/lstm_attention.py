# Description: LSTM model
# Author: ZhouJH
# Data: 2020/4/8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append(".")
from model.focal_loss import FocalLoss
from model.registry import MODELS

loss_dict = {"CELoss": nn.CrossEntropyLoss(),
             "FocalLoss": FocalLoss(4)}


@MODELS.register_module
class LSTM_ATTENTION(nn.Module):

    def __init__(self,
                 input_size,
                 cls_num,
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.5,
                 fc_size=512,
                 loss_func='CELoss',
                 with_cuda=True):

        super(LSTM_ATTENTION, self).__init__()
        self.input_size = input_size
        self.cls_num = cls_num
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if with_cuda:
            self.init_w = Variable(torch.zeros(1, hidden_size), requires_grad=True).cuda()
        else:
            self.init_w = Variable(torch.zeros(1, hidden_size), requires_grad=True)
        self.init_w = nn.Parameter(self.init_w)

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

        assert loss_func in loss_dict.keys()
        self.loss_func = loss_dict[loss_func]

    def _attention_layer(self, lstm_out):
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out = torch.tanh(lstm_out) # shape (batch_size, seq_len, hidden_size)
        M = torch.matmul(self.init_w, lstm_out.permute(0,2,1))
        # print("M size = {}".format(m.size()))
        alpha = F.softmax(M, dim=0) # shape (batch_size, 1, seq_len)
        out = torch.matmul(alpha, lstm_out).squeeze() # shape (batch_size, hidden_size)
        return out

    def forward(self, x):
        out, (h_n, h_c) = self.rnn_layer(x)
        out = self._attention_layer(out)
        out = self.fc_layer1(out)
        out = self.fc_layer2(out)
        return out

    def loss(self, out, label):
        return self.loss_func(out, label)
