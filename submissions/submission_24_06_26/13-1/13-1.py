import torch
import torch.distributed as dist

# Initialize torch.distributed with the MPI backend
dist.init_process_group(backend='mpi')

# Print the rank and size on every process
rank = dist.get_rank()
size = dist.get_world_size()

###############
# BLOCKING
###############

# Allocate a 3 x 4 tensor on each rank
if rank == 0:
    tensor = torch.ones(3, 4)
else:
    tensor = torch.zeros(3, 4)

# Use blocking sends and receives to send rank 0's tensor to rank 1
if rank == 0:
    dist.send(tensor, dst=1)
elif rank == 1:
    dist.recv(tensor, src=0)

# Check if the tensor on rank 1 is the same as the tensor on rank 0
if rank == 1:
    print(f"Tensor on rank 1 after receiving:")
    print(tensor)
    
###############
# NON-BLOCKING
###############

# Allocate a 3 x 4 tensor on each rank
if rank == 0:
    tensor = torch.ones(3, 4)
else:
    tensor = torch.zeros(3, 4)

# Use blocking sends and receives to send rank 0's tensor to rank 1
if rank == 0:
    dist.isend(tensor, dst=1)
elif rank == 1:
    dist.irecv(tensor, src=0)

# Check if the tensor on rank 1 is the same as the tensor on rank 0
if rank == 1:
    print(f"Tensor on rank 1 after i-receiving:")
    print(tensor)
    
###############
# ALLREDUCE
###############

# Allocate a 3 x 4 tensor and initialize it
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

# Perform an allreduce with the reduce operation SUM on the tensor
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# Print the result
print(f"Allreduce result on rank {rank}:")
print(tensor)

# Clean up torch.distributed
dist.destroy_process_group()