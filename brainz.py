import torch 
import torch.nn as nn
import random

class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
    def get_outputs(self, data):
        self.eval()  
        with torch.no_grad():
            outputs = self(data)
        return outputs
    
    # def evaluate(self, input):
    #     adjusted = torch.tensor(input, dtype=torch.float32)
    #     out = self(adjusted)
    #     return self.map_output(out)
    
    # def map_output(self, out):
    #     #check monday
    #     mapped = int(out) % 8
    #     return mapped
    
    def mutate(self, mutation_rate = .1, mutation_magnitude = 1):
        for param in self.parameters():
            if(random.random() <= mutation_rate):
                param.data += mutation_magnitude * torch.rand_like(param)