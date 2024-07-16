import torch

ten6List = [[[0, 2, 4], 
             [6, 8, 10]], 
            
            [[1, 3, 5], 
             [7, 9, 11]]]

ten1 = torch.tensor(ten6List)
print(ten1.size())