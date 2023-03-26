import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import copy
import random
import numpy as np
import sys

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class DNN_proto(nn.Module):
    def __init__(self, input_shape=1 * 28 * 28, mid_dim=100, num_classes=10):
        try:
            super(DNN_proto, self).__init__()

            self.fc0 = nn.Linear(input_shape, mid_dim)
            self.fc = nn.Linear(mid_dim, num_classes)
        except Exception as e:
            print("DNN_proto_2")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def forward(self, x):
        try:
            x = torch.flatten(x, 1)
            rep = F.relu(self.fc0(x))
            x = self.fc(rep)
            output = F.log_softmax(x, dim=1)
            return output, rep
        except Exception as e:
            print("DNN_proto_2 forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


class DNN(nn.Module):
    def __init__(self, input_shape=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



class CNN(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          32,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32,
                          64,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.fc1 = nn.Sequential(
                nn.Linear(mid_dim*4, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            print("CNN")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc(out)
            return out
        except Exception as e:
            print("CNN forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


class CNN_proto(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_proto, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
                          32,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32,
                          64,
                          kernel_size=5,
                          padding=0,
                          stride=1,
                          bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.fc1 = nn.Sequential(
                nn.Linear(mid_dim * 4, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            print("CNN proto")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            rep = self.fc1(out)
            out = self.fc(rep)
            return out, rep
        except Exception as e:
            print("CNN proto forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


class Logistic(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        out = F.log_softmax(x, dim=1)
        return out


class Logistic_Proto(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Logistic_Proto, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        rep = self.fc(x)
        out = F.log_softmax(rep, dim=1)
        return out, rep
