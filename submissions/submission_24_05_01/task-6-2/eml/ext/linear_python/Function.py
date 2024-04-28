import torch

class ownFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        with torch.no_grad():
            out = input.mm(weight.t())
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        with torch.no_grad():
            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
            
        return grad_input, grad_weight
