import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.head(x)
        
        
class PPO(nn.Module):
    def __init__(self, action_size, depth):
        super(PPO, self).__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(depth, 16, kernel_size=8, stride=4, padding=4)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=2)
        #self.bn2 = nn.BatchNorm2d(64)
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        #self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(4608, 256)
        self.head = nn.Linear(256, action_size+1)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        #x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        
        probs = F.softmax(x[:,:self.action_size] - torch.max(x[:,:self.action_size],1)[0].unsqueeze(1))
        val = x[:,-1]
        
        return probs, val
