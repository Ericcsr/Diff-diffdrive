import torch
import torch.nn as nn


class Experiment:
    def __init__(self,env,agent):
        self.env   = env
        self.agent = agent
        
    def rollout_train(self):
        done = False
        total_loss = 0
        self.env.reset()
        while not done:
            _,_,loss,_,done = self.env.step(self.agent)
            total_loss += loss
        return total_loss

    def rollout_test(self):
        done = False
        while not done:
            _,_,_,o2,done = self.env.step(self.agent)
            self.env.render(o2)