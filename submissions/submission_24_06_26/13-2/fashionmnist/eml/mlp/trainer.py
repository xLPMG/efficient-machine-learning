import torch

## Trains the given MLP-model.
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied (single epoch).
#  @param io_model model which is trained.
#  @param io_optimizer.
#  @return summed loss over all training samples.

def train( i_loss_func,
           io_data_loader,
           io_model,
           io_optimizer,
           i_size_distributed):
  # switch model to training mode
  io_model.train()

  l_loss_total = 0
  
  for instances, labels in io_data_loader:
      io_optimizer.zero_grad()

      predictions = io_model(instances)
      loss = i_loss_func(predictions, labels)
        
      loss.backward()
        
      # reduce gradients
      for l_pa in io_model.parameters():
            torch.distributed.all_reduce( l_pa.grad.data,
                  op = torch.distributed.ReduceOp.SUM )
            l_pa.grad.data = l_pa.grad.data / float(i_size_distributed)
  
      io_optimizer.step()
      l_loss_total += loss.item()

  return l_loss_total
