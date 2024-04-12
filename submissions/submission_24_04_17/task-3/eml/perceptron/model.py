import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x.float()
        x = self.linear(x)
        x = self.sigmoid(x)
        return x