import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self, input_shape):
        self.hidden = F.leaky_relu
        self.input_shape = input_shape
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=4)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2, padding=3)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=2)
        #self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        #self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.n_size = self.conv_output(self.input_shape) 
        
        self.fc1 = nn.Linear(self.n_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size()[0], self.n_size)
        x = self.hidden(self.fc1(x))
        x = self.hidden(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def forward_conv(self, x):
        x = self.pool(self.hidden(self.conv1(x)))
        x = self.hidden(self.conv2(x))
        x = self.hidden(self.conv3(x))
        #x = self.hidden(self.conv4(x))
        #x = self.hidden(self.conv5(x))
        return x
        
    def conv_output(self, shape):
        inp = Variable(torch.rand(1, *shape))
        out_f = self.forward_conv(inp)
        n_size = out_f.data.view(1, -1).size(1)
        return n_size

