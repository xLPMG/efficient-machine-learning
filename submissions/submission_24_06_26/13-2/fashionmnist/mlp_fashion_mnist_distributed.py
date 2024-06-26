import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.distributed as dist
import time

dist.init_process_group(backend='mpi')
rank = dist.get_rank()
size = dist.get_world_size()

from pathlib import Path
Path("out").mkdir(exist_ok=True)

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

# start timer
if rank == 0:
    start = time.time()
    print("Started timer")
    print("Training on ", size, " processes")
    
# Create data loaders
batch_size = 64
seed = 123456789

dist_train_sampler = DistributedSampler(training_data, num_replicas=size, rank=rank, shuffle=True, drop_last=True, seed=seed)
dist_test_sampler = DistributedSampler(test_data, num_replicas=size, rank=rank, shuffle=True, drop_last=True, seed=seed)

train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=dist_train_sampler)
test_dataloader = DataLoader(test_data, batch_size=batch_size, sampler=dist_test_sampler)

import eml.mlp.trainer as trainer
import eml.mlp.tester as tester
import eml.mlp.model as model
import eml.vis.fashion_mnist as visMNIST

# set seed for reproducibility
torch.manual_seed(123456789)
    
my_eml_model = model.Model()
loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.SGD(my_eml_model.parameters(), lr=0.05)

epochs = 26
visualize = False

for epoch in range(epochs):
    total_loss = trainer.train(loss_func, train_dataloader, my_eml_model, l_optimizer, size)
    total_loss = torch.as_tensor(total_loss)
    test_loss, num_correct = tester.test(loss_func, test_dataloader, my_eml_model)
    test_loss = torch.as_tensor(test_loss)
    num_correct = torch.as_tensor(num_correct)
    
    torch.distributed.all_reduce( total_loss, op = torch.distributed.ReduceOp.SUM )
    total_loss = total_loss / float(size)
    
    torch.distributed.all_reduce( test_loss, op = torch.distributed.ReduceOp.SUM )
    test_loss = test_loss / float(size)
    
    torch.distributed.all_reduce( num_correct, op = torch.distributed.ReduceOp.SUM )
    
    if rank == 0:
        print(f"Epoch {epoch}/{epochs-1}, Total Loss: {total_loss}, Test loss: {test_loss}, Correct samples: {num_correct}")
        if epoch % 5 == 0 and visualize == True:
            visMNIST.plot(0, 1000, test_dataloader, my_eml_model, f"out/vis_{epoch}.pdf")

if rank == 0:   
    end = time.time()
    duration = end - start
    print("Finished training for ", epochs, " epochs. Duration: ", duration, " seconds.")
 
dist.destroy_process_group()