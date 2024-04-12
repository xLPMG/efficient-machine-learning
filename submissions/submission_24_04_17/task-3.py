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
    
#############################################################
# TASK 3-3
#############################################################

## Trains the given linear perceptron.
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader which provides the training data.
#  @param io_model model which is trained.
#  @param io_optimizer used optimizer.
#  @return loss.
def train( i_loss_func,
           io_data_loader,
           io_model,
           io_optimizer ):
    # switch model to training mode
    io_model.train()

    l_loss_total = 0

    for inputs, labels in io_data_loader:
        io_optimizer.zero_grad()
        # forward pass
        outputs = io_model(inputs)
        # calculate loss
        loss = i_loss_func(outputs, labels)
        # backward pass and optimization
        loss.backward()
        io_optimizer.step()
        # update total loss
        l_loss_total += loss.item()

    return l_loss_total

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
model = Model()
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = train(loss_func, data_loader, model, optimizer)
    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss}")

