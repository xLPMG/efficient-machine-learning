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
print(l_tensor_complex_view)
print("size: ", l_tensor_complex_view.size())
print("stride: ", l_tensor_complex_view.stride())

# [::2,1,:] slices the tensor along the three axis: z, y and x.
# -> ::2 slices along the z-axis: from start to end with a step 
#    size of 2, meaning that every second element is selected
# -> 1 slices along the y-axis: only the element at index 1 
#    is selected (effectively the second row)
# -> : slices along the x-axis: every element is selected
# In summary this means that we select T0 and T2, and for each of those
# we select the second row (since y = 1) and keep all x entries.

# We are left with 2 rows of 3 elements each, which explains a size of (2, 3)
# Since we kept the rows intact, the stride in x direction stays 1.
# Going from 3 to 15, we jump over 12 elements in the original tensor.
# Another explanation is that instead of jumping only 3 elements 
# (like in the original tensor), we now skip an additional 3 rows,
# resulting in 12 elements total.

##########################################################
# TASK 5
##########################################################
print("############### TASK 4 ###############")

l_tensor_complex_view = l_tensor_complex_view.contiguous()
print(l_tensor_complex_view)
print("size: ", l_tensor_complex_view.size())
print("stride: ", l_tensor_complex_view.stride())

# .contiguous() returns a contiguous in memory tensor 
# containing the same data as l_tensor_complex_view.
# Since the data is now contiguous in memory, the 
# stride is what we would naively expect when looking
# at the view our our tensor: 1 in x-direction and
# 3 in y-direction, since we skip 3 elements in
# the memory to jump between rows