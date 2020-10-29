import torch
import torch.nn as nn
import numpy as np

class DiffDrive(nn.Module):
    def __init__(self,sim_time,l,r,initial_pose):
        super(DiffDrive,self).__init__()
        ## Constants which doesn't requires gradient
        # sim_time must be small enought
        self.sim_time = sim_time
        self.l        = l
        self.r        = r
        ## Variables which needs gradients
        # x = [x,y,theta]
        self.x     = torch.tensor(initial_pose[0],dtype=torch.float32,requires_grad=True)
        self.y     = torch.tensor(initial_pose[1],dtype=torch.float32,requires_grad=True)
        self.theta = torch.tensor(initial_pose[2],dtype=torch.float32,requires_grad=True)
        self.v     = 0
        self.omega = 0

    ## Simply implement Explicity integration which can be handled by AD in pytorch.
    # May be need to change to acceleration control after ward.
    def step(self,a):
        self.v     = (self.r*a[0] + self.r*a[1])/2
        self.omega = (self.r*a[0] - self.r*a[1])/(2*self.l)
        theta_new  = self.theta + self.omega
        self.x     = self.x + self.v*torch.cos((self.theta+theta_new)/2)
        self.y     = self.x + self.v*torch.sin((self.theta+theta_new)/2)
        self.theta = theta_new

    def get_obs(self):
        return torch.tensor([self.x,self.y,self.theta,
                             self.v,self.omega]).float()
