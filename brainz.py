import torch 
import torch.nn as nn
import random
from configz import CONFIG

class simpleNN(nn.Module):
    def __init__(self):
        super(simpleNN, self).__init__()
        self.fc = nn.Linear(12, 8)
        self.fc2 = nn.Linear(8, 6)
    
    def forward(self, x):
        x = self.fc(x)
        #x = torch.relu(x)
        x = torch.tanh(x)
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
    
    def mutate(self, mutation_rate=CONFIG["mutation_rate"], mutation_magnitude=CONFIG["mutation_magnitude"]):
        for param in self.parameters():
            if(random.random() <= mutation_rate):
                param.data += mutation_magnitude * (2 * torch.rand_like(param) - 1)