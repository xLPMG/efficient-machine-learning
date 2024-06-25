from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from SimpleDataSet import SimpleDataSet
import torch.distributed as dist

dist.init_process_group(backend='mpi')
rank = dist.get_rank()
size = dist.get_world_size()

dataset = SimpleDataSet( 16 )
batch_size = 3
drop_last = False

# DistributedSampler
# 
# dataset                           Dataset used for sampling.
# num_replicas  (int, optional)     Number of processes participating in distributed training.
# rank          (int, optional)     Rank of the current process within num_replicas.
# shuffle       (bool, optional)    If True (default), sampler will shuffle the indices.
# seed          (int, optional)     random seed used to shuffle the sampler if shuffle=True. (identical across processes)
# drop_last     (bool, optional)    If True, then the sampler will drop the tail of the data to make it evenly divisible across the number of replicas. 
#                                   If False, the sampler will add extra indices to make the data evenly divisible across the replicas.
dist_sampler = DistributedSampler(dataset, num_replicas=size, rank=rank, shuffle=True, drop_last=drop_last)

print(rank,"/",size-1,": DataLoader", list(DataLoader(dataset, sampler=dist_sampler, batch_size=batch_size, drop_last=drop_last)))
print(rank,"/",size-1,": BatchSampler", list(BatchSampler(dist_sampler, batch_size=batch_size, drop_last=drop_last)))

dist.destroy_process_group()