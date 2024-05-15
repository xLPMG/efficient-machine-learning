from triton_matmul import matmul
import torch
import time

run_time = 120 # minimum run time in seconds

def benchmark(N, dtype):
    a = torch.randn((N, N), device='cuda', dtype=dtype)
    b = torch.randn((N, N), device='cuda', dtype=dtype)
    c1 = torch.empty((N, N), device='cuda', dtype=dtype)
    c2 = torch.empty((N, N), device='cuda', dtype=dtype)

    # Benchmark torch implementation
    elapsed_time_ms = 0
    repetitions = 0
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    
    start1.record()
    while elapsed_time_ms < run_time*1000:   
        c1 += torch.matmul(a, b)
        repetitions+=1
        end1.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start1.elapsed_time(end1)
        
    torch_flops = 2 * (N ** 3) * repetitions
    torch_ops = torch_flops*1000 / elapsed_time_ms
    torch_tflops = torch_ops / (10 ** 12)
    
    # Benchmark triton implementation
    elapsed_time = 0
    repetitions = 0
    
    start2 = time.time()
    while elapsed_time < run_time:
        c2 += matmul(a, b)
        repetitions+=1
        elapsed_time = time.time() - start2
        
    triton_flops = 2 * (N ** 3) * repetitions
    triton_ops = triton_flops / elapsed_time
    triton_tflops = triton_ops / (10 ** 12)        
    
    print("========================================================")
    print("Completed benchmark:")
    print(" Matrix size:    ",N," * ",N)
    print(" Data type:      ",dtype)
    print(" TFLOP/s torch:  ", torch_tflops)
    print(" TFLOP/s triton: ", triton_tflops)

print("##########################################################")
print("FLOAT32")
print("##########################################################")
benchmark(32, torch.float32)
benchmark(256, torch.float32)
benchmark(4096, torch.float32)
benchmark(8192, torch.float32)

print("##########################################################")
print("FLOAT16")
print("##########################################################")
benchmark(32, torch.float16)
benchmark(256, torch.float16)
benchmark(4096, torch.float16)
benchmark(8192, torch.float16)