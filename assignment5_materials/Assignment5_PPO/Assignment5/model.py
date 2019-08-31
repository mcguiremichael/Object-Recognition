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
        self.conv1 = nn.Conv2d(depth, 32, kernel_size=8, stride=4, padding=4)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(9216, 512)
        self.head = nn.Linear(512, action_size+1)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        
        probs = F.softmax(x[:,:self.action_size] - torch.max(x[:,:self.action_size],1)[0].unsqueeze(1))
        val = x[:,-1]
        
        return probs, val
    
class PPO_LSTM(nn.Module):
    def __init__(self, action_size, hidden_size=64, depth=1):
        super(PPO_LSTM, self).__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(depth, 32, kernel_size=8, stride=4, padding=4)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(hidden_size+64, 64, kernel_size=3, stride=1, padding=1)
        #self.bn3 = nn.BatchNorm2d(64)
        
        self.lstm = ConvLSTM(64, hidden_size)
        self.fc = nn.Linear(9216, 512)
        self.head = nn.Linear(512, action_size+1)
        
    def forward(self, x, hidden):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        output, hidden = self.lstm(x, hidden)
        x = torch.cat([x, output], dim=1)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        
        probs = F.softmax(x[:,:self.action_size] - torch.max(x[:,:self.action_size],1)[0].unsqueeze(1))
        val = x[:,-1]
        
        return probs, val, hidden
    
    def forward_unroll(self, x, hiddens, unroll_steps=1):
        s = x.shape
        flat_x = x.flatten(0,1)
        features = self.forward_features(flat_x)
        features = features.view((s[:2]) + (features.shape[1:]))
        for i in range(unroll_steps):
            outputs, hiddens = self.lstm(features[:,i], hiddens)
        x = torch.cat([features[:,-1], outputs], dim=1)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        
        probs = F.softmax(x[:,:self.action_size] - torch.max(x[:,:self.action_size],1)[0].unsqueeze(1))
        val = x[:,-1]
        
        return probs, val, hiddens
        
    def forward_features(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return x
    
    def init_hidden(self, batch_size=1, device="cuda:0"):
        return self.lstm.init_hidden_state(batch_size, width=12, use_torch=True, device=device)


"""
    Convolutional implementation of LSTM.
    Check https://pytorch.org/docs/stable/nn.html#lstm to see reference information
"""
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvLSTM, self).__init__()

        self.input_size, self.hidden_size = input_size, hidden_size

        self.input_to_input = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.hidden_to_input = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)

        self.input_to_forget = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.hidden_to_forget = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)

        self.input_to_gate = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.hidden_to_gate = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)

        self.input_to_output = nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.hidden_to_output = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)

        self.inpad = nn.ReflectionPad2d(1)
        self.hpad = nn.ReflectionPad2d(1)



    """
        input: torch tensor of shape (N, D, H, W)
        hidden_state: torch tensor of shape (N, 2, D, H, W)
        D is placeholder as usual
    """
    def forward(self, input, hidden_state):

        h_0 = hidden_state[:,0]
        c_0 = hidden_state[:,1]
        input = self.inpad(input)
        h_0 = self.hpad(h_0)

        i = F.sigmoid(self.input_to_input(input) + self.hidden_to_input(h_0))
        f = F.sigmoid(self.input_to_forget(input) + self.hidden_to_forget(h_0))
        g = F.tanh(self.input_to_gate(input) + self.hidden_to_gate(h_0))
        o = F.sigmoid(self.input_to_output(input) + self.hidden_to_output(h_0))
        c_t = f * c_0 + i * g
        h_t = o * F.tanh(c_t)

        hidden_state_out = torch.cat([h_t.unsqueeze(1), c_t.unsqueeze(1)], dim=1)

        return o, hidden_state_out

    def init_hidden_state(self, batch_size=1, width=8, use_torch=True, device="cuda:0"):
        if (use_torch):
            return torch.zeros((batch_size, 2, self.hidden_size, width, width)).float().to(device)
        else:
            return np.zeros((batch_size, 2, self.hidden_size, width, width)).astype(np.float32)
