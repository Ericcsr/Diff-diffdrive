import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,obs_dim,act_dim):
        self.fc1 = nn.Linear(obs_dim,32)
        self.fc2 = nn.Linear(32,32)
        self.out = nn.Linear(32,act_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DiffsimAgent:
    # initiaized with parameters
    def __init__(self,obs_dim,act_dim,lr=0.01):
        self.net = Net(obs_dim,act_dim)
        self.optim = torch.optim.Adam(self.net.parameters(),lr=lr)

    # take action:
    def act(self,o):
        act = self.net(o)
        # May be need to clip the action
        return act

    # Update the net
    def update(self,loss):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    
