import torch

class ownFunc( torch.autograd.function ):
    
    def __init__( self,
              i_n_features_input,
              i_n_features_output ):
    
    @staticmethod
    def forward(input, weight):
        with torch.no_grad():
            out = torch.matmul(input, weight)
        return out
    
    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight = inputs
        ctx.save_for_backward(input, weight)
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        with torch.no_grad:
            grad_input = ...
            grad_weight = ...
            
        return grad_input, grad_weight
        
results = ownFunc.apply(l_x, l_w.transpose(0,1))
results.backward(l_grad)