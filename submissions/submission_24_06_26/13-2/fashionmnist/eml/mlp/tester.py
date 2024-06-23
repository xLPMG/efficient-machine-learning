import torch

## Tests the model
#  @param i_loss_func used loss function.
#  @param io_data_loader data loader containing the data to which the model is applied.
#  @param io_model model which is tested.
#  @return summed loss over all test samples, number of correctly predicted samples.

def test( i_loss_func,
          io_data_loader,
          io_model ):
  # switch model to evaluation mode
  io_model.eval()
  
  l_loss_total = 0
  l_n_correct = 0
  
  with torch.no_grad():
        for instances, labels in io_data_loader:
            predictions = io_model(instances)
            l_loss_total += i_loss_func(predictions, labels).item()
            l_n_correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()

  return l_loss_total, l_n_correct
