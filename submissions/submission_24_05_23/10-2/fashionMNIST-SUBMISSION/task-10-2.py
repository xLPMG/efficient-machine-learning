# external imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# internal imports
import eml.mlp.trainer as trainer
import eml.mlp.tester as tester
import eml.mlp.model as model
import eml.vis.fashion_mnist as visMNIST

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

# # Visualize some images
# fig, axes = plt.subplots(2, 5, figsize=(10, 4))
# for ax, (image, label) in zip(axes.flatten(), training_data):
#     ax.imshow(image.squeeze(), cmap="gray")
#     ax.set_title(label)
#     ax.axis("off")

# with PdfPages('out/image_visualizations.pdf') as pdf:
#     pdf.savefig(fig)
# plt.close()
    
# Create data loaders
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# set seed for reproducibility
torch.manual_seed(123456789)
    
my_eml_model = model.Model()
loss_func = torch.nn.CrossEntropyLoss()
l_optimizer = torch.optim.SGD(my_eml_model.parameters(), lr=0.05)

# START OF AIMET QUANTIZATION
import aimet_common.defs
import aimet_torch.quantsim

def calibrate(io_model, args, i_use_cuda=False):
    dataloader = args
    io_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for inputs, _ in dataloader:
            if i_use_cuda:
                inputs = inputs.cuda()
            io_model(inputs)

dummy_input = torch.rand(1, 1, 28, 28)
sim = aimet_torch.quantsim.QuantizationSimModel(
    my_eml_model, 
    quant_scheme=aimet_common.defs.QuantScheme.post_training_tf, 
    dummy_input=dummy_input, 
    default_param_bw=8, 
    default_output_bw=8
)

sim.compute_encodings(
    forward_pass_callback=calibrate, 
    forward_pass_callback_args=train_dataloader
)

# END OF AIMET QUANTIZATION

epochs = 26
visualize = False

for epoch in range(epochs):
    total_loss = trainer.train(loss_func, train_dataloader, sim.model, l_optimizer)
    test_loss, num_correct = tester.test(loss_func, test_dataloader, sim.model)
    print(f"Epoch {epoch}/{epochs-1}, Total Loss: {total_loss}, Test loss: {test_loss}, Correct samples: {num_correct}")
    if epoch % 5 == 0 and visualize == True:
        visMNIST.plot(0, 1000, test_dataloader, my_eml_model, f"out/vis_{epoch}.pdf")