import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#############################################################
# TASK 4-1
#############################################################
print("################# TASK 4-1 #################")

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, (image, label) in zip(axes.flatten(), training_data):
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(label)
    ax.axis("off")

with PdfPages('image_visualizations.pdf') as pdf:
    pdf.savefig(fig)
    
    
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#############################################################
# TASK 4-2
#############################################################
print("################# TASK 4-2 #################")

import eml.mlp.trainer as trainer
import eml.mlp.tester as tester
import eml.mlp.model as model

# set seed for reproducibility
torch.manual_seed(123456789)
    
my_eml_model = model.Model()
loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.SGD(my_eml_model.parameters(), lr=0.05)

epochs = 20

for epoch in range(epochs):
    total_loss = trainer.train(loss_func, train_dataloader, my_eml_model, l_optimizer)
    test_loss, num_correct = tester.test(loss_func, test_dataloader, my_eml_model)
    
    print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss}, Test loss: {test_loss}, Correct samples: {num_correct}")
    
