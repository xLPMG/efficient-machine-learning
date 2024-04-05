import torch

P = [[0, 1, 2], [3, 4, 5]]
Q = [[6, 7, 8], [9, 10, 11]]

tenP = torch.tensor(P)
tenQ = torch.tensor(Q)
tenQT = torch.transpose(tenQ, 0, 1)

##########################################################
# TASK 1
##########################################################
print("############### TASK 1 ###############")

print("P + Q")
print("=====================")
print("using torch.add:")
print(torch.add(tenP, tenQ))
print("using +:")
print(tenP + tenQ)
# Observation: entries of the same position are added
# -> element-wise addition
# e.g 0+6=6, 1+7=8, 2+8=10

print("P * Q")
print("=====================")
print("using torch.mul:")
print(torch.mul(tenP, tenQ))
print("using *:")
print(tenP * tenQ)
# Observation: this form of matrix multiplication 
# simply multiplies entries of the same position
# -> element-wise multiplication
# e.g 0*6=0, 1*7=7, 2*8=16

# In both cases it can be seen that the overloaded
# binary operators provide the same result as the
# pytorch functions

##########################################################
# TASK 2
##########################################################
print("############### TASK 2 ###############")

print("matrix multiplication of P and Q^T")
print("=====================================")
print("using torch.matmul:")
print(torch.matmul(tenP, tenQT))
print("using @:")
print(tenP @ tenQT)

# It can be seen that the overloaded binary operator
# provides the same result as the pytorch function.

##########################################################
# TASK 3
##########################################################
print("############### TASK 3 ###############")

# Torch.sum will sum up all matrix elements into one value
print("torch.sum")
print("=====================================")
print("P   :",torch.sum(tenP))
print("Sum of Q and Q^T should be the same:")
print("Q   :", torch.sum(tenQ))
print("Q^T :", torch.sum(tenQT))

# Torch.max will output the greatest element of the matrix
print("torch.max")
print("=====================================")
print("P   :",torch.max(tenP))
print("Max element of Q and Q^T should be the same:")
print("Q   :", torch.max(tenQ))
print("Q^T :", torch.max(tenQT))

##########################################################
# TASK 4
##########################################################
print("############### TASK 4 ###############")

# In the first snippet, l_tmp is merely assigned to reference 
# l_tensor_0. Both point to the same location in memory. 
# In the second snippet, l_tmp is an actual and separate copy of 
# l_tensor_1. The difference is that modifiying l_tmp in the first
# snippet will also modify l_tensor_0, however modifiying l_tmp in 
# the second snippet will leave l_tensor_1 untouched. Demonstration:

l_tensor_0 = torch.tensor([[0, 1, 2], [3, 4, 5]])
l_tensor_1 = torch.tensor([[6, 7, 8], [9, 10, 11]])

print("-------- SNIPPET 1 --------")
l_tmp = l_tensor_0
print("l_tensor_0 before l_tmp is modified:")
print(l_tensor_0)

l_tmp[:] = 0

print("l_tensor_0 after l_tmp is modified:")
print(l_tensor_0)

print("-------- SNIPPET 2 --------")
l_tmp = l_tensor_1.clone().detach()
print("l_tensor_1 before l_tmp is modified:")
print(l_tensor_1)

l_tmp[:] = 0

print("l_tensor_1 after l_tmp is modified:")
print(l_tensor_1)

