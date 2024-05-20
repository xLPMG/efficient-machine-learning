import torch
import torchvision.models as models

# Load a pretrained ResNet model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Define a simple transform to preprocess the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the CIFAR10 dataset
calibration_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=True)

from aimet_torch.quantsim import QuantizationSimModel

def forward_pass(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            model(data)

# Create QuantizationSimModel
sim = QuantizationSimModel(model, dummy_input=torch.randn(1, 3, 224, 224))

# Set quantization configurations
sim.compute_encodings(forward_pass, calibration_loader)

# Perform evaluation to check the accuracy of the quantized model
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

# Example evaluation
test_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False)  # Use test set in real scenarios
evaluate(sim.model, test_loader)

# Save the quantized model
sim.export(path='export', filename_prefix='quantized_resnet18', dummy_input=torch.randn(1, 3, 224, 224))

