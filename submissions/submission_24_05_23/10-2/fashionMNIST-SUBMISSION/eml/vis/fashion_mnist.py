import torch
import matplotlib.pyplot as plt

## Converts an Fashion MNIST numeric id to a string.
#  @param i_id numeric value of the label.
#  @return string corresponding to the id.
def toLabel( i_id ):
  l_labels = [ "T-Shirt",
               "Trouser",
               "Pullover",
               "Dress",
               "Coat",
               "Sandal",
               "Shirt",
               "Sneaker",
               "Bag",
               "Ankle Boot" ]

  return l_labels[i_id]

## Applies the model to the data and plots the data.
#  @param i_off offset of the first image.
#  @param i_stride stride between the images.
#  @param io_data_loader data loader from which the data is retrieved.
#  @param io_model model which is used for the predictions.
#  @param i_path_to_pdf optional path to an output file, i.e., nothing is shown at runtime.
def plot( i_off,
          i_stride,
          io_data_loader,
          io_model,
          i_path_to_pdf = None ):
    
  # switch to evaluation mode
  io_model.eval()

  # create pdf if required
  if( i_path_to_pdf != None ):
    import matplotlib.backends.backend_pdf
    l_pdf_file = matplotlib.backends.backend_pdf.PdfPages( i_path_to_pdf )

  for instances, labels in io_data_loader:
    predictions = io_model(instances)
    
    # loop using offset and stride
    for j in range(i_off, len(instances), i_stride):
      instance = instances[j]
      label = labels[j]
      prediction = predictions[j]

      # convert label and prediction to strings
      label_str = toLabel(label.item())
      # select the highest prediction
      prediction_str = toLabel(torch.argmax(prediction).item())
      
      plt.imshow(instance.squeeze(), cmap='gray')
      plt.title(f'Predicted: {prediction_str}, Actual: {label_str}')
      plt.axis('off')
      if( i_path_to_pdf != None ):
        l_pdf_file.savefig()
      else:
        plt.show()
      plt.close()

  # close pdf if required
  if( i_path_to_pdf != None ):
    l_pdf_file.close()
  