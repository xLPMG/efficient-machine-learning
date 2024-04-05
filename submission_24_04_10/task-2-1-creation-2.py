import torch

# declare dimensions
zDim = 4
yDim = 2
xDim = 3

tensorList = []
# since the values are just consecutively increased numbers, 
# we can use a counter variable
i = 0

# for each of the T_i,
# add the list of rows
for z in range(zDim):
    # t_y contains yDim t_x-lists (rows)
    t_y = []
    for y in range(yDim):
        # create a list for a new row 
        # and fill it with xDim values
        t_x = []
        for x in range(xDim):
            t_x.append(i)
            i += 1
        t_y.append(t_x)
    tensorList.append(t_y)
        
tensor = torch.tensor(tensorList)
print(tensor)