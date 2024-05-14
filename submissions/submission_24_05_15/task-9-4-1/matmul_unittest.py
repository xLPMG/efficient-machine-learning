from triton_matmul import matmul
import torch

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
    print("✅ Triton and Torch match: fp16")
else:
    print("❌ Triton and Torch differ: fp16")
    
a = torch.randn((512, 512), device='cuda', dtype=torch.float32)
b = torch.randn((512, 512), device='cuda', dtype=torch.float32)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp32_inputs={triton_output}")
print(f"torch_output_with_fp32_inputs={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-1):
    print("✅ Triton and Torch match: fp32")
else:
    print("❌ Triton and Torch differ: fp32")