import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmarks = True
import math
import numpy as np

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
    
class PPO_MHDPA(nn.Module):
    def __init__(self, action_size, depth, device="cuda:0"):
        super(PPO_MHDPA, self).__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(depth+2, 16, kernel_size=5, stride=2, padding=2)
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=2)
        #self.bn2 = nn.BatchNorm2d(64)
        self.attentionBlock1 = nn.Sequential(
            RelationalModule(
                32, 16, 2
            ),
            RelationalModule(
                32, 16, 2
            ),
            nn.MaxPool2d(2)
        )
        self.attentionBlock2 = nn.Sequential(
            RelationalModule(
                32, 32, 2
            ),
            RelationalModule(
                64, 32, 2
            )
        )
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=2)
        
        
        #self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(2304, 512)
        self.head = nn.Linear(512, action_size+1)
        
        width = 84
        self.coords = torch.from_numpy(np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, width))[0]).float().unsqueeze(0).unsqueeze(0)
        self.coords = torch.cat([self.coords, self.coords.transpose(-2,-1)], dim=1).to(device)
        self.coords.requires_grad = False
        
        
    def forward(self, x):
        local_coords = self.coords.expand((len(x), 2) + (x.shape[2:]))
        x = torch.cat([local_coords, x], dim=1)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.attentionBlock1(x)
        x = self.attentionBlock2(x)
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc(x))
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
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=2)
        #self.bn3 = nn.BatchNorm2d(64)
        
        self.lstm = ConvLSTM(64, hidden_size, width=7)
        self.conv4 = ResnetBlock(hidden_size, 2)
        self.fc = nn.Linear(49*hidden_size, 512)
        self.head = nn.Linear(512, action_size+1)
        
    def forward(self, x, hidden):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        output, hidden = self.lstm(x, hidden)
        x = self.conv4(output)
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
        x = self.conv4(outputs)
        x = F.leaky_relu(self.fc(x.view(x.size(0), -1)))
        x = self.head(x)
        
        probs = F.softmax(x[:,:self.action_size] - torch.max(x[:,:self.action_size],1)[0].unsqueeze(1))
        val = x[:,-1]
        
        return probs, val, hiddens
        
    def forward_features(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        return x
    
    def init_hidden(self, batch_size=1, device="cuda:0"):
        return self.lstm.init_hidden_state(batch_size, width=7, use_torch=True, device=device)


"""
    Convolutional implementation of LSTM.
    Check https://pytorch.org/docs/stable/nn.html#lstm to see reference information
"""
"""
    Convolutional implementation of LSTM.
    Check https://pytorch.org/docs/stable/nn.html#lstm to see reference information
"""
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, width=8, device="cuda:0"):
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

        self.Wc_input = Variable(torch.zeros(1, hidden_size, width, width)).to(device)
        self.Wc_forget = Variable(torch.zeros(1, hidden_size, width, width)).to(device)
        self.Wc_output = Variable(torch.zeros(1, hidden_size, width, width)).to(device)

        self.Wc_input.requires_grad = True
        self.Wc_forget.requires_grad = True
        self.Wc_output.requires_grad = True


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

        i = F.sigmoid(self.input_to_input(input) + self.hidden_to_input(h_0) + c_0 * self.Wc_input)
        f = F.sigmoid(self.input_to_forget(input) + self.hidden_to_forget(h_0) + c_0 * self.Wc_forget)
        g = F.tanh(self.input_to_gate(input) + self.hidden_to_gate(h_0))
        c_t = f * c_0 + i * g
        o = F.sigmoid(self.input_to_output(input) + self.hidden_to_output(h_0) + c_t * self.Wc_output)
        h_t = o * F.tanh(c_t)

        hidden_state_out = torch.cat([h_t.unsqueeze(1), c_t.unsqueeze(1)], dim=1)

        return o, hidden_state_out

    def init_hidden_state(self, batch_size=1, width=8, use_torch=True, device="cuda:0"):
        if (use_torch):
            return torch.zeros((batch_size, 2, self.hidden_size, width, width)).float().to(device)
        else:
            return np.zeros((batch_size, 2, self.hidden_size, width, width)).astype(np.float32)
        
"""
    Convolutional implementation of LSTM.
    Check https://pytorch.org/docs/stable/nn.html#lstm to see reference information
"""
class ConvLSTM_res(nn.Module):
    def __init__(self, input_size, hidden_size, width=8, device="cuda:0"):
        super(ConvLSTM_res, self).__init__()

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

        self.Wc_input = Variable(torch.zeros(1, hidden_size, width, width)).to(device)
        self.Wc_forget = Variable(torch.zeros(1, hidden_size, width, width)).to(device)
        self.Wc_output = Variable(torch.zeros(1, hidden_size, width, width)).to(device)

        self.Wc_input.requires_grad = True
        self.Wc_forget.requires_grad = True
        self.Wc_output.requires_grad = True
        

    """
        input: torch tensor of shape (N, D, H, W)
        hidden_state: torch tensor of shape (N, 2, D, H, W)
        D is placeholder as usual
    """
    def forward(self, input, hidden_state):

        h_0 = hidden_state[:,0]
        c_0 = hidden_state[:,1]
        alt_input = self.inpad(input)
        alt_h_0 = self.hpad(h_0)

        i = F.sigmoid(self.input_to_input(alt_input) + self.hidden_to_input(alt_h_0) + c_0 * self.Wc_input)
        f = F.sigmoid(self.input_to_forget(alt_input) + self.hidden_to_forget(alt_h_0) + c_0 * self.Wc_forget)
        g = F.tanh(self.input_to_gate(alt_input) + self.hidden_to_gate(alt_h_0))
        c_t = c_0 + f * c_0 + i * g
        o = F.sigmoid(self.input_to_output(alt_input) + self.hidden_to_output(alt_h_0) + c_t * self.Wc_output)
        h_t = h_0 + o * F.tanh(c_t)

        hidden_state_out = torch.cat([h_t.unsqueeze(1), c_t.unsqueeze(1)], dim=1)

        return o, hidden_state_out

    def init_hidden_state(self, batch_size=1, width=8, use_torch=True, device="cuda:0"):
        if (use_torch):
            return torch.zeros((batch_size, 2, self.hidden_size, width, width)).float().to(device)
        else:
            return np.zeros((batch_size, 2, self.hidden_size, width, width)).astype(np.float32)


class RelationalProjection(nn.Module):
    def __init__(self, in_size, num_features, device="cuda:0"):

        super(RelationalProjection, self).__init__()
        self.linear = nn.Linear(in_size, num_features)
        self.norm = nn.InstanceNorm1d(num_features)
        self.function = nn.ReLU()
        
        self.gamma = Variable(torch.ones((1,1,num_features)).float()).to(device)
        self.beta = Variable(torch.zeros((1,1,num_features)).float()).to(device)

    def forward(self, x):
        x = self.linear(x)
        x = self.gamma * self.norm(x) + self.beta
        return self.function(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_size, num_features, num_heads, device="cuda:0"):

        super(SelfAttentionBlock, self).__init__()
        self.device = device
        self.in_size = in_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.QueryLayers = []
        self.KeyLayers = []
        self.ValueLayers = []

        for i in range(in_size):
            self.QueryLayers.append(RelationalProjection(in_size, num_features).to(self.device))
            self.KeyLayers.append(RelationalProjection(in_size, num_features).to(self.device))
            self.ValueLayers.append(RelationalProjection(in_size, num_features).to(self.device))
        self.MLP = nn.Sequential(
            nn.Linear(self.num_heads * self.num_features, self.num_heads * self.num_features),
            nn.ReLU(),
            nn.Linear(self.num_heads * self.num_features, self.num_heads * self.num_features),
            nn.ReLU()
        ).to(self.device)

        self.output_norm = nn.InstanceNorm1d(num_features)
        self.gamma = Variable(torch.ones((1,1,num_features*num_heads)).float()).to(device)
        self.beta = Variable(torch.zeros((1,1,num_features*num_heads)).float()).to(device)
        

    def forward(self, x):
        (N, D, H, W) = x.shape
        new_x = x.permute(0, 2, 3, 1)
        flattened = new_x.flatten(start_dim=1, end_dim=-2)

        heads_out = []

        for i in range(self.num_heads):
            Q = self.QueryLayers[i](flattened)
            K = self.KeyLayers[i](flattened)
            V = self.ValueLayers[i](flattened)
            numerator = torch.matmul(Q, K.permute(0,2,1))
            scaled = numerator / math.sqrt(self.num_features)
            attention_weights = F.softmax(scaled)
            A = torch.matmul(attention_weights, V)
            heads_out.append(A)

        heads_out = torch.cat(heads_out, dim=-1)
        output = self.output_norm(self.MLP(heads_out) + heads_out)
        output = self.gamma * output + self.beta
        output = output.permute(0,2,1).contiguous().view((N, -1, H, W))

        return output
    
class RelationalModule(nn.Module):

    def __init__(self, in_size, num_features, num_heads, encode=True):
        super(RelationalModule, self).__init__()
        self.in_size=in_size
        self.num_features=num_features
        self.num_heads=num_heads
        self.encode = encode
        self.net_encoding = nn.Conv2d(in_size, num_heads*num_features, kernel_size=1, stride=1, padding=0)
        self.mhdpa = nn.TransformerEncoderLayer(num_heads*num_features, num_heads, dim_feedforward=num_features)

    def forward(self, x):
        if (self.encode and self.in_size != self.num_heads*self.num_features):
            x = self.net_encoding(x)
        (N, D, H, W) = x.shape
        new_x = x.permute(0, 2, 3, 1)
        new_x = new_x.flatten(start_dim=1, end_dim=-2)
        output = self.mhdpa(new_x)
        output = output.permute(0,2,1).contiguous().view((N, -1, H, W))
        return output


class ResnetBlock(nn.Module):
    def __init__(self, num_features, num_layers):
        super(ResnetBlock, self).__init__()

        self.activation = nn.ReLU
        self.residuals = nn.Sequential()
        for i in range(num_layers):
            self.residuals.add_module(
                "residual" + str(i+1),
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                    #nn.BatchNorm2d(num_features),
                    self.activation()
                )
            )

    def forward(self, x):
        out = self.residuals(x)
        return x + out
