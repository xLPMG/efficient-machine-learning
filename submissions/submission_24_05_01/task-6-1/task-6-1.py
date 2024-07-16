import torch
from torch import nn

# TASK 6-1-2
l_linear_torch = nn.Linear(in_features=3, out_features=2, bias=False)
l_linear_torch.weight.data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad = True)
l_input_data = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad = True)

# TASK 6-1-3
l_output = l_linear_torch(l_input_data)
print("Forward pass result:")
print(l_output)

# TASK 6-1-4
l_output.backward(torch.ones_like(l_output))
print("Gradients w.r.t. the input:")
print(l_input_data.grad)
print("Gradients w.r.t. the weights:")
print(l_linear_torch.weight.grad)