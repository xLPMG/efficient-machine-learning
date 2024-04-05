import torch

##########################################################
# TASK 1
##########################################################
print("############### TASK 1 ###############")

# CODE FROM task-2-1-creation-2.py:
zDim = 4
yDim = 2
xDim = 3
tensorList = []
i = 0
for z in range(zDim):
    t_y = []
    for y in range(yDim):
        t_x = []
        for x in range(xDim):
            t_x.append(i)
            i += 1
        t_y.append(t_x)
    tensorList.append(t_y)
tensor = torch.tensor(tensorList)

print("tensor  :", tensor)
print("size    :", tensor.size())
print("stride  :", tensor.stride())
print("dtype   :", tensor.dtype)
print("layout  :", tensor.layout)
print("device  :", tensor.device)

##########################################################
# TASK 2 & 3
##########################################################
print("############### TASK 2 & 3 ###############")

l_tensor_float = torch.tensor(tensorList, dtype=torch.float32)
l_tensor_fixed = l_tensor_float[:,0,:]

print("========= l_tensor_float =========")
print("tensor  :", l_tensor_float)
print("size    :", l_tensor_float.size())
print("stride  :", l_tensor_float.stride())
print("dtype   :", l_tensor_float.dtype)
print("layout  :", l_tensor_float.layout)
print("device  :", l_tensor_float.device)

print("========= l_tensor_fixed =========")
print("tensor  :", l_tensor_fixed)
print("size    :", l_tensor_fixed.size())
print("stride  :", l_tensor_fixed.stride())
print("dtype   :", l_tensor_fixed.dtype)
print("layout  :", l_tensor_fixed.layout)
print("device  :", l_tensor_fixed.device)

# Observations
# From l_tensor_float to l_tensor_fixed, the second 
# dimension was removed, and thus size and stride lost a parameter. 
# dtype, layout and device stayed the same.

##########################################################
# TASK 4
##########################################################
print("############### TASK 4 ###############")

l_tensor_complex_view = l_tensor_float[::2,1,:]

