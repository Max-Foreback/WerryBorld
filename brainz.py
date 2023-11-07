import torch 
import torch.nn as nn
import random

#torch.manual_seed(4)

class simpleNN(nn.Module):
    def __init__(self, ins):
        super(simpleNN, self).__init__()
        self.fc = nn.Linear(ins, 1)
    
    def forward(self, x):
        return self.fc(x)
    
    def eval(self, input):
        adjusted = torch.tensor(input, dtype=torch.float32)
        out = self(adjusted)
        return self.map_output(out)
    
    def map_output(self, out):
        #check monday
        mapped = int(out) % 8
        return mapped
    
    def mutate(self, mutation_rate = .1, mutation_magnitude = 1):
        for param in self.parameters():
            if(random.random() <= mutation_rate):
                param.data += mutation_magnitude * torch.rand_like(param)