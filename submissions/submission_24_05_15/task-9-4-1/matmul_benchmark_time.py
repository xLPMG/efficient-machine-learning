from triton_matmul import matmul
import torch
import time

def benchmark(M, N, K, num_runs, dtype):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    c1 = torch.empty((M, N), device='cuda', dtype=dtype)
    c2 = torch.empty((M, N), device='cuda', dtype=dtype)

    # Benchmark torch implementation
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)
    
    start1.record()
    for x in range(num_runs):   
        c1 += torch.matmul(a, b)
        torch.cuda.synchronize()
    end1.record()
    
    # Benchmark triton implementation
    start2 = time.time()
    for x in range(num_runs):  
        c2 += matmul(a, b)
    end2 = time.time()
    
    print("========================================================")
    print("Completed benchmark:")
    print(" Matrix sizes:")
    print("     M:",M)
    print("     N:",N)
    print("     K:",K)
    print(" Data type:      ",dtype)
    print(" Number of runs: ", num_runs)
    print(" Time torch:     ", start1.elapsed_time(end1)/1000,"seconds")
    print(" Time triton:    ", end2-start2,"seconds")

print("##########################################################")
print("FLOAT32")
print("##########################################################")
benchmark(32, 32, 32, 1000, torch.float32)
benchmark(256, 256, 256, 1000, torch.float32)
benchmark(4096, 4096, 4096, 1000, torch.float32)
benchmark(8192, 8192, 8192, 1000, torch.float32)

print("##########################################################")
print("FLOAT16")
print("##########################################################")
benchmark(32, 32, 32, 1000, torch.float16)
benchmark(256, 256, 256, 1000, torch.float16)
benchmark(4096, 4096, 4096, 1000, torch.float16)
benchmark(8192, 8192, 8192, 1000, torch.float16)