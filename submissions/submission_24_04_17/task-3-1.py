import torch
import numpy as np
import matplotlib.pyplot as plt

#############################################################
# TASK 3-1
#############################################################

points = np.genfromtxt("data_points.csv", delimiter=",")
points = torch.from_numpy(points)

labels = np.genfromtxt("data_labels.csv", delimiter=",")
labels = torch.from_numpy(labels)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

zero_points = points[labels == 0]
ax.scatter(zero_points[:, 0], zero_points[:, 1], zero_points[:, 2], color='black')

one_points = points[labels == 1]
ax.scatter(one_points[:, 0], one_points[:, 1], one_points[:, 2], color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#############################################################
# TASK 3-2
#############################################################

data_set = torch.utils.data.TensorDataset(points, labels)
data_loader = torch.utils.data.DataLoader (data_set, batch_size = 64)

for p, l in data_loader:
    print(p)
    print(l)
