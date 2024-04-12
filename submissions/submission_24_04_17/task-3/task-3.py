import torch
import numpy as np
import matplotlib.pyplot as plt

#############################################################
# TASK 3-1
#############################################################
print("################# TASK 3-1 #################")

points = np.genfromtxt("data_points.csv", delimiter=",")
points = torch.from_numpy(points).float()

labels = np.genfromtxt("data_labels.csv", delimiter=",")
labels = torch.from_numpy(labels).float()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

zero_points = points[labels == 0]
one_points = points[labels == 1]

# select all rows (:) and a specific column (0, 1, or 2) for the dimension
ax.scatter(zero_points[:, 0], zero_points[:, 1], zero_points[:, 2], color='black')
ax.scatter(one_points[:, 0], one_points[:, 1], one_points[:, 2], color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
print("(showing plot...)")
plt.show()

#############################################################
# TASK 3-2
#############################################################
print("################# TASK 3-2 #################")

#unsqueeze for task 3-3
labels = torch.unsqueeze( labels, -1 )

data_set = torch.utils.data.TensorDataset(points, labels)
data_loader = torch.utils.data.DataLoader (data_set, batch_size = 64)

# The batch loader should give us 64 samples in each iteration,
# and the rest (<=64) in the last iteration.
print("Printing point and label sizes for each iteration:")
batch_num = 0
for p, l in data_loader:
    print("Points, batch #", batch_num, ": ", p.size())
    print("Labels, batch #", batch_num, ": ", l.size())
    batch_num+=1
    
# It should print "Points, batch #x:  torch.Size([64, 3])" for points, because we have 64 samples with 3 dimensions each per batch
# It should print "Labels, batch #x: torch.Size([64, 1])" for labels, because we have 64 samples in only one dimension per batch
    
#############################################################
# TASK 3-3
#############################################################
print("################# TASK 3-3 #################")

import eml.perceptron.trainer as trainer
import eml.perceptron.model as model

# set seed for reproducibility
torch.manual_seed(123456789)
    
my_eml_model = model.Model()
loss_func = torch.nn.BCELoss()
l_optimizer = torch.optim.SGD(my_eml_model.parameters(), lr=0.05)

epochs = 50

for epoch in range(epochs):
    total_loss = trainer.train(loss_func, data_loader, my_eml_model, l_optimizer)
    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss}")

