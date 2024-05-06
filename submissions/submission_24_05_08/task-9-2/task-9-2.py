import torch
import time
import matplotlib.pyplot as plt

def run_benchmark(fileName):
    file = open(fileName, "w")
    file.write("DEVICE,SIZE,DTYPE,REPETITIONS,TIME,GFLOPS,TFLOPS")
    file.close()
    
    # CONFIGURATION
    sizes = [512, 1024, 2048, 4096, 8192, 10240]
    dtypes = [torch.float64, torch.float32, torch.bfloat16, torch.float16]
    run_time = 10 # minimum run time in seconds
    
    # CPU TEST
    for size in sizes:
        for dtype in dtypes:
            
            tensor1 = torch.rand(size, size, dtype = dtype, device = torch.device("cpu"))
            tensor2 = torch.rand(size, size, dtype = dtype, device = torch.device("cpu"))
            tensor3 = torch.rand(size, size, dtype = dtype, device = torch.device("cpu"))

            elapsed_time = 0
            repetitions = 0
            
            start_time = time.time()
            while elapsed_time < run_time:
                tensor3 += tensor1 @ tensor2
                repetitions+=1
                elapsed_time = time.time() - start_time

            flops = 2 * (size ** 3) * repetitions
            ops = flops / elapsed_time
            gflops = ops / (10 ** 9)
            tflops = ops / (10 ** 12)
            
            file = open(fileName, "a")
            file.write("\ncpu,%i,%s,%i,%f,%f,%f" % (size,dtype,repetitions,elapsed_time,gflops,tflops))
            file.close()
            
    # GPU TEST
    torch.backends.cuda.matmul.allow_tf32 = False
    for size in sizes:
        for dtype in dtypes:
            
            tensor1 = torch.rand(size, size, dtype = dtype)
            tensor2 = torch.rand(size, size, dtype = dtype)
            tensor3 = torch.rand(size, size, dtype = dtype)
            
            tensor1 = tensor1.cuda()
            tensor2 = tensor2.cuda()
            tensor3 = tensor2.cuda()
            
            elapsed_time_ms = 0
            repetitions = 0
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            while elapsed_time_ms < run_time*1000:
                tensor3 += tensor1 @ tensor2
                repetitions+=1
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                
            flops = 2 * (size ** 3) * repetitions
            ops = flops*1000 / elapsed_time_ms
            gflops = ops / (10 ** 9)
            tflops = ops / (10 ** 12)
            
            file = open(fileName, "a")
            file.write("\ngpu,%i,%s,%i,%f,%f,%f" % (size,dtype,repetitions,elapsed_time_ms/1000,gflops,tflops))
            file.close()
            
    # TF32 GPU TEST
    torch.backends.cuda.matmul.allow_tf32 = True
    for size in sizes:
        tensor1 = torch.rand(size, size, dtype = dtype)
        tensor2 = torch.rand(size, size, dtype = dtype)
        tensor3 = torch.rand(size, size, dtype = dtype)
        
        tensor1 = tensor1.cuda()
        tensor2 = tensor2.cuda()
        tensor3 = tensor2.cuda()
        
        elapsed_time_ms = 0
        repetitions = 0
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        while elapsed_time_ms < run_time*1000:
            tensor3 += tensor1 @ tensor2
            repetitions+=1
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            
        flops = 2 * (size ** 3) * repetitions
        ops = flops*1000 / elapsed_time_ms
        gflops = ops / (10 ** 9)
        tflops = ops / (10 ** 12)   
            
        file = open(fileName, "a")
        file.write("\ngpu,%i,TF32,%i,%f,%f,%f" % (size,repetitions,elapsed_time_ms/1000,gflops,tflops))
        file.close()

# RUN BENCHMARK
output_file = "results.csv"

start_time = time.time()
print("RUNNING GEMM BENCHMARK")
run_benchmark(output_file)
end_time = time.time()
print("Total time for benchmark: ", end_time-start_time,"seconds")