from numpy import load
import torch

data = load('seismic/labels_train.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item].dtype)
    print(torch.as_tensor(data[item]).shape)