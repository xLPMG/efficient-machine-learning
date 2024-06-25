import torch
import torch.distributed as dist

# Initialize torch.distributed with the MPI backend
dist.init_process_group(backend='mpi')

# Print the rank and size on every process
rank = dist.get_rank()
size = dist.get_world_size()
print(f"Rank: {rank}, Size: {size}")

###############
# BLOCKING
###############

# Allocate a 3 x 4 tensor on each rank
if rank == 0:
    tensor1 = torch.ones(3, 4)
else:
    tensor1 = torch.zeros(3, 4)

# Use blocking sends and receives to send rank 0's tensor to rank 1
if rank == 0:
    dist.send(tensor1, dst=1)
elif rank == 1:
    dist.recv(tensor1, src=0)

# Check if the tensor on rank 1 is the same as the tensor on rank 0
if rank == 1:
    print(f"Tensor1 on rank 1 after receiving:", tensor1)
    
###############
# NON-BLOCKING
###############

# Allocate a 3 x 4 tensor on each rank
if rank == 0:
    tensor2 = torch.ones(3, 4)
else:
    tensor2 = torch.zeros(3, 4)

# Use blocking sends and receives to send rank 0's tensor to rank 1
if rank == 0:
    request = dist.isend(tensor2, dst=1)
    request.wait()
elif rank == 1:
    request = dist.irecv(tensor2, src=0)
    request.wait()

# Check if the tensor on rank 1 is the same as the tensor on rank 0
if rank == 1:
    print(f"Tensor2 on rank 1 after i-receiving:", tensor2)
    
dist.barrier()

###############
# ALLREDUCE
###############

# Allocate a 3 x 4 tensor and initialize it
tensor3 = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

# Perform an allreduce with the reduce operation SUM on the tensor
dist.all_reduce(tensor3, op=dist.ReduceOp.SUM)

# Print the result
print(f"Allreduce result on rank {rank}:", tensor3)

# Clean up torch.distributed
dist.barrier()
dist.destroy_process_group()
