import torch

print("###################################")
print("torch.zeros")
print("###################################")

# The following code will produce matrices of given size that are filled with ones
tenZ1 = torch.zeros(2, 2)
print("size: 2 x 2")
print(tenZ1)

tenZ2 = torch.zeros(2, 4)
print("size: 2 x 4")
print(tenZ2)

tenZ3 = torch.zeros(4, 2)
print("size: 4 x 2")
print(tenZ3)

print("###################################")
print("torch.ones")
print("###################################")

# The following code will produce matrices of given size that are filled with ones
tenO1 = torch.ones(2, 2)
print("size: 2 x 2")
print(tenO1)

tenO2 = torch.ones(2, 4)
print("size: 2 x 4")
print(tenO2)

tenO3 = torch.ones(4, 2)
print("size: 4 x 2")
print(tenO3)

print("###################################")
print("torch.ones_like")
print("###################################")
# this function will take an existing tensor and create another one filled with ones with the same size.
print("New tensor:")
tenOL1 = torch.ones_like(tenZ2)
print(tenOL1)

print("Proof that old tensor was not modified:")
print(tenZ2)

print("###################################")
print("torch.rand")
print("###################################")

# The following code will produce random matrices of given size.
# The matrices should have random values, even though the same 
# function is called
print(torch.rand(2, 3))
print(torch.rand(2, 3))
print(torch.rand(2, 3))