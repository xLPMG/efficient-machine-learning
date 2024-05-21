import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# the model which was trained in submission_24_04_24
# I saved it using "torch.save(my_eml_model, "fashionmnistmodel")"
my_eml_model = torch.load("fashionmnistmodel")

# Download test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for instances, labels in loader:            
            scores = model(instances)
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
    
    model.train()

# check the accuracy of the model before quantization
check_accuracy(test_dataloader, my_eml_model)

# START OF AIMET QUANTIZATION
import aimet_common.defs
import aimet_torch.quantsim

def calibrate(io_model, args, i_use_cuda=False):
    dataloader = args
    batch_size = dataloader.batch_size
    if i_use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    io_model.eval()
    samples = 1000
    batch_cntr = 0
    with torch.no_grad():
        for input_data, _ in dataloader:

            inputs_batch = input_data.to(device)
            io_model(inputs_batch)
            batch_cntr += 1
            if (batch_cntr * batch_size) > samples:
                break

dummy_input = torch.rand(1, 1, 28, 28) # input shape for FashionMNIST
sim = aimet_torch.quantsim.QuantizationSimModel(
    my_eml_model, 
    quant_scheme=aimet_common.defs.QuantScheme.post_training_tf, 
    dummy_input=dummy_input, 
    default_param_bw=8, 
    default_output_bw=8
)

sim.compute_encodings(
    forward_pass_callback=calibrate, 
    forward_pass_callback_args=test_dataloader
)

# END OF AIMET QUANTIZATION

# check the accuracy of the model after quantization
check_accuracy(test_dataloader, sim.model)

# export
sim.export("exported_model", filename_prefix="fashionmnist-quantized", dummy_input=dummy_input)
torch.save(sim.model, "exported_model/fashionmnist-quantized-torchmodel")