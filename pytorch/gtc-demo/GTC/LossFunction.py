# -*- coding: utf-8 -*-

from torch import nn
import torch
from torch.nn import functional as F

class FocalLoss(nn.Module):

    def __init__(self, gamma=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def margin(cos, label, m, s):
    #m = 0.35
    #s = 30.
    phi = cos - m
    label = label.view(-1, 1)
    index = cos.data * 0.0
    index.scatter_(1, label.data.view(-1, 1), 1)
    index = index.byte()
    output = cos * 1.0
    output[index] = phi[index]
    output *= s
    return output
