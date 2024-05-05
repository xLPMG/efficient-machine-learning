import torch
from torch import nn
import torch.version

if not torch.cuda.is_available:
    print("CUDA was not found but is required for this application.")
    exit()

print("############## TASK 9-1-1 ##############")
# create tensor on CPU
tensor = torch.tensor([ [-0.5, 0.2],
                        [0.3, -0.4] ])
# offload to GPU
tensor = tensor.cuda()
print(tensor)
print( "Perform ReLU" )
tensor = nn.ReLU(inplace=True)(tensor)
print(tensor)

print("############## TASK 9-1-2 ##############")
# create tensors on CPU
tensor1 = torch.tensor([[-0.5, 0.2],
                        [0.3, -0.4]])
tensor2 = torch.tensor([[-0.4, 0.3],
                        [0.2, -0.5]])

# change dtype
tensor1 = tensor1.to(torch.bfloat16)
tensor2 = tensor2.to(torch.bfloat16)

# move to GPU
tensor1 = tensor1.cuda()
tensor2 = tensor2.cuda()

print(tensor1)
print("t1 dtype: ", tensor1.dtype)

print(tensor2)
print("t2 dtype: ", tensor2.dtype)

# operation
tensor3 = tensor1 @ tensor2
print("Result of tensor1 @ tensor2")
print(tensor3)

print("############## TASK 9-1-3 ##############")

# the exponent for thr following FP operations
# will be 2^(16-15) = 2

print("----------------- FP16 -----------------")
# 10-bit mantissa
# => create an FP32 tensor where the 11th bit is set,
# then convert to FP16 and compare the values.
# => setting tensor to 2^-11 should result in different values

# 2^-10
in_bounds = torch.tensor( 2 * 1.0009765625,
                          dtype = torch.float32 )

# 2^-11
overflow = torch.tensor( 2 * 1.0004882812,
                         dtype = torch.float32 )

print("in bounds FP32 tensor:\n", in_bounds)
print("FP16 conversion:\n", in_bounds.to( torch.float16 ))

print("overflow FP32 tensor:\n", overflow)
print("FP16 conversion:\n", overflow.to( torch.float16 ))

print("----------------- BFP16 ----------------")
# 7-bit mantissa
# => create an FP32 tensor where the 8th bit is set,
# then convert to BFP16 and compare the values.
# => setting tensor to 2^-11 should result in different values

# 2^-7
in_bounds = torch.tensor( 2 * 1.0078125,
                          dtype = torch.float32 )

# 2^-8
overflow = torch.tensor( 2 * 1.00390625,
                         dtype = torch.float32 )

print("in bounds FP32 tensor:\n", in_bounds)
print("FP16 conversion:\n", in_bounds.to( torch.bfloat16 ))

print("overflow FP32 tensor:\n", overflow)
print("BFP16 conversion:\n", overflow.to( torch.bfloat16 ))

print("----------------- TF32 -----------------")
# TF32 is a little different, as it is only used on the GPU
# and during operations. I will therefore compare a
# FP32 @ FP32 with a TF32 @ TF32 operation.

# The TF32 mantissa is 10 bits, so I will work with
# tensors that have the 11th bit set (2^-11)

tensor1 = torch.tensor( [[2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812],
                         [2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812],
                         [2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812],
                         [2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812]],
                        dtype = torch.float32 )

tensor2 = torch.tensor( [[2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812],
                         [2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812],
                         [2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812],
                         [2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812, 2 * 1.0004882812]],
                        dtype = torch.float32 )

print("tensor1: \n", tensor1)
print("tensor2: \n", tensor2)

# move to GPU
tensor1 = tensor1.cuda()
tensor2 = tensor1.cuda()

# disable TF32 on GPU
torch.backends.cuda.matmul.allow_tf32 = False

result = tensor1 @ tensor2
print("FP32 @ FP32: \n", result) 

# enable TF32 on GPU
torch.backends.cuda.matmul.allow_tf32 = True

result = tensor1 @ tensor2
print("TF32 @ TF32: \n", result) 