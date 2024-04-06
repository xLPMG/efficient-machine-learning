import torch
import ctypes

##########################################################
# TASK 1
##########################################################
print("############### TASK 1 ###############")

# START CODE FROM task-2-1-creation-2.py:
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
# END CODE FROM task-2-1-creation-2.py

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
print("############### TASK 5 ###############")

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

##########################################################
# TASK 6
##########################################################
print("############### TASK 6 ###############")

# For this task I will use a simpler tensor than before:

ten6List = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
ten6 = torch.tensor(ten6List, dtype=torch.float32)
print(ten6)

# The strides are (6, 3, 1)
print(ten6.stride())
zStride = 6
yStride = 3
xStride = 1

# A pointer to the start our tensor can be retrieved using
ten6Ptr = ten6.data_ptr()

################################################################
# In the following section, I will access different elements
# of the tensor using pointer arithmatic

# size of element in bytes
eSize = 4

# first element
out = (ctypes.c_float).from_address( ten6Ptr )
print("First value       :", out)

# jump one element in x-direction
out = (ctypes.c_float).from_address( ten6Ptr + 1 * eSize * xStride)
print("Jump in x-dir     :", out)

# jump one element in y-direction
out = (ctypes.c_float).from_address( ten6Ptr + 1 * eSize * yStride )
print("Jump in y-dir     :", out)

# jump one element in x and y-direction
out = (ctypes.c_float).from_address( ten6Ptr + 1 * eSize * xStride + 1 * eSize * yStride )
print("Jump in x-y-dir   :", out)

# jump in z-direction
out = (ctypes.c_float).from_address( ten6Ptr + 1 * eSize * zStride)
print("Jump in z-dir     :", out)

# jump one element in x-y-z-direction
out = (ctypes.c_float).from_address( ten6Ptr + 1 * eSize * xStride + 1 * eSize * yStride + 1 * eSize * zStride)
print("Jump in x-y-z-dir :", out)

# this works because in memory, the tensor is represented as
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
# The strides tell us how many elements we need to skip in
# memory to perform a jump in tensor view.
# For each of those elements, we need to increase the address
# by eSize, which is 4 in our example since a float32 value 
# takes up 4 Bytes of memory.