import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
import SimpleDataSet

dataset = SimpleDataSet()

# DistributedSampler
# 
# dataset                           Dataset used for sampling.
# num_replicas  (int, optional)     Number of processes participating in distributed training.
# rank          (int, optional)     Rank of the current process within num_replicas.
# shuffle       (bool, optional)    If True (default), sampler will shuffle the indices.
# seed          (int, optional)     random seed used to shuffle the sampler if shuffle=True. (identical across processes)
# drop_last     (bool, optional)    If True, then the sampler will drop the tail of the data to make it evenly divisible across the number of replicas. 
#                                   If False, the sampler will add extra indices to make the data evenly divisible across the replicas.
dist_sampler = DistributedSampler(dataset, num_replicas=4, rank=2, shuffle=True, drop_last=False)
dataloader = DataLoader(dataset, sampler=dist_sampler)
print(list(dataloader))

# Wrap into BatchSampler
print(list(BatchSampler(dist_sampler, batch_size=3, drop_last=False)))