import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self, input_shape):
        self.hidden = F.relu
        self.input_shape = input_shape
        super(Classifier, self).__init__()
        CONV1 = 16
        CONV2 = 20
        CONV3 = 20
        CONV4 = 20
        self.conv1 = nn.Conv2d(3, CONV1, 7, stride=2, padding=3)
        self.conv1_bn = nn.BatchNorm2d(CONV1)
        self.conv2 = nn.Conv2d(CONV1, CONV2, 5, stride=2, padding=2)
        self.conv2_bn = nn.BatchNorm2d(CONV2)
        self.conv3 = nn.Conv2d(CONV2, CONV3, 4, stride=2, padding=2)
        self.conv3_bn = nn.BatchNorm2d(CONV3)
        self.conv4 = nn.Conv2d(CONV3, CONV4, 4, stride=2, padding=2)
        self.conv4_bn = nn.BatchNorm2d(CONV4)
        
        #self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        #self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.n_size = self.conv_output(self.input_shape) 
        
        FC1 = 256
        FC2 = 256
        self.fc1 = nn.Linear(self.n_size, FC1)
        self.fc1_bn = nn.BatchNorm1d(FC1)
        #self.fc2 = nn.Linear(FC1, FC2)
        #self.fc2_bn = nn.BatchNorm1d(FC2)
        self.fc2 = nn.Linear(FC1, NUM_CLASSES)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size()[0], self.n_size)
        x = self.dropout(self.hidden(self.fc1_bn(self.fc1(x))))
        #x = self.dropout(self.hidden(self.fc2_bn(self.fc2(x))))
        x = self.fc2(x)
        return x
        
    def forward_conv(self, x):
        x = self.hidden(self.conv1_bn(self.conv1(x)))
        x = self.hidden(self.conv2_bn(self.conv2(x)))
        x = self.hidden(self.conv3_bn(self.conv3(x)))
        x = self.hidden(self.conv4_bn(self.conv4(x)))
        #x = self.hidden(self.conv4(x))
        #x = self.hidden(self.conv5(x))
        return x
        
    def conv_output(self, shape):
        inp = Variable(torch.rand(1, *shape))
        out_f = self.forward_conv(inp)
        n_size = out_f.data.view(1, -1).size(1)
        return n_size

