# MPS implementation for Apple Silicon Mac (Metal)
import torch
from torch import nn

print("############## TASK 9-1-1 ##############")
if torch.backends.mps.is_available():
    print( "MPS is available" )
    mps_device = torch.device("mps")
    
    # create tensor on CPU
    tensor = torch.tensor([[-0.5, 0.2],
                            [0.3, -0.4]])
    # offload to GPU
    tensor = tensor.to(mps_device)
    print(tensor)
    print( "Perform ReLU" )
    tensor = nn.ReLU(inplace=True)(tensor)
    print(tensor)
else:
    print( "Error: MPS is not available" )
    
print("############## TASK 9-1-2 ##############")
if torch.backends.mps.is_available():
    print( "MPS is available" )
    mps_device = torch.device("mps")
    
    # create tensors on CPU
    tensor1 = torch.tensor([[-0.5, 0.2],
                            [0.3, -0.4]])
    tensor2 = torch.tensor([[-0.5, 0.2],
                            [0.3, -0.4]])
    
    # move to GPU
    tensor1 = tensor1.to(mps_device)
    tensor2 = tensor2.to(mps_device)
    
    # change dtype - BF16 NOT POSSIBLE: TypeError: BFloat16 is not supported on MPS
    tensor1.to(torch.float16)
    tensor2.to(torch.float16)
    
    print(tensor1)
    print("dtype: ", tensor1.dtype, ", device: ",tensor1.device)
    
    print(tensor2)
    print("dtype: ", tensor2.dtype, ", device: ",tensor2.device)
    
    # operation
    tensor3 = tensor1 @ tensor2
    print("Result of tensor1 @ tensor2")
    print(tensor3)

else:
    print( "Error: MPS is not available" )
    