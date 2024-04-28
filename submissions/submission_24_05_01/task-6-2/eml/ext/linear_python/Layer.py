import torch
import Function
from torch import nn

class ownLayer(nn.Module):
    def __init__( self,
                  i_n_features_input,
                  i_n_features_output ):
        super().__init__()
        self.input_features = i_n_features_input
        self.output_features = i_n_features_output
        self.weight = nn.Parameter(torch.empty(i_n_features_output, i_n_features_input))

    def forward(self, input):
        return Function.apply(input, self.weight)