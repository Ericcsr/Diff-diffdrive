import torch
import torch.nn as nn
import numpy as np
from motion_models import DiffDrive
from diffsim_agent import DiffsimAgent
import matplotlib.pyplot as plt

class TargetNaiveEnv:
    ## Initialize the env
    def __init__(self,sim_time,initial_pose,l,r,max_len):
        self.sim_time    = sim_time
        self.initial_pose= initial_pose
        self.l           = l
        self.r           = r
        self.t     = 0
        #     Initial Version can only use x,y as goal.
        self.goal  = torch.tensor([0,0]).float()
        self.obs_dim = 8
        self.act_dim = 2
        self.max_len = 120
        self.steps    = 0
        self.figure = plt.figure()
        self.reset()

    ## Compute loss: Need to deal with ring reference problem
    def compute_loss(self):
        loss = 0
        loss += (self.goal[0] - self.sim.x)**2
        loss += (self.goal[1] - self.sim.y)**2
        return loss

    ##  set the target from interface.
    def set_target(self,x,y):
        self.goal[0] = x
        self.goal[1] = y

    ## reset the diffsim
    def reset(self):
        self.sim = DiffDrive(sim_time    = self.sim_time,
                             l           = self.l,
                             r           = self.r,
                             initial_pose= self.initial_pose)
        self.t = 0

    def step(self,agent):
        o = self.sim.get_obs()
        o = torch.cat([o,self.goal,torch.tensor(self.t).float])
        a = agent.act(o.data)
        a = torch.clamp(a,max=2,min=-2)
        self.sim.step(a)
        o2= self.sim.get_obs()
        loss = self.compute_loss()
        self.t += self.sim_time
        self.steps += 1
        done  = False
        if self.steps >= self.max_len:
            done = True
        return o,a,loss,o2,done

    def render(self,o):
        x     = float(o[0])
        y     = float(o[1])
        theta = float(o[2])
        plt.clf()
        plt.xlim([-10,10])
        plt.ylim([-10,10])
        circle = plt.Circle((x,y),0.5,fill=False)
        plt.gcf().add_artist(circle)
        plt.plot([x,x+0.4*np.cos(theta)],
                 [y,y+0.4*np.sin(theta)])
        plt.plot(self.goal[0],self.goal[1],'ro')
        plt.axes().set_aspect('equal')
        plt.pause(0.01)

