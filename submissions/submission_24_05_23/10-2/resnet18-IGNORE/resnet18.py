import os
import torch   
from torchvision.models import resnet18
model = resnet18(pretrained=True)

from aimet_torch.model_preparer import prepare_model
model = prepare_model(model)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))

from aimet_torch.batch_norm_fold import fold_all_batch_norms
_ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))

from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda:
    dummy_input = dummy_input.cuda()
sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

print(sim.model)
print(sim)

def pass_calibration_data(sim_model, use_cuda):
    data_loader = ImageNetDataPipeline.get_val_dataloader()
    batch_size = data_loader.batch_size

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()
    samples = 1000

    batch_cntr = 0
    with torch.no_grad():
        for input_data, target_data in data_loader:

            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)

            batch_cntr += 1
            if (batch_cntr * batch_size) > samples:
                break

sim.compute_encodings(forward_pass_callback=pass_calibration_data,
                      forward_pass_callback_args=use_cuda)

accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)
print(accuracy)
