import numpy as np
import csv

classProbsFile = "class_probs.raw"
pathResult_CPUFP32 = "output/cpu_fp32/output/Result_"
pathResult_CPUI8 = "output/cpu_int8/Result_"
pathResult_HOSTI8 = "output/host_int8/Result_"
pathResult_GPUI8 = "output/gpu_int8/Result_"
pathResult_HTPI8 = "output/htp_int8/output/Result_"
pathLabel = "/opt/data/imagenet/raw_test/batch_size_32/labels_"

def read_raw(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        class_probs = np.frombuffer(raw_data, dtype=np.float32)
        return class_probs
    
def read_csv(file_path):
    results = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            results.append(row)
    return results

for i in range(10):
    # Calculates probabilities for each class
    calculated_probs_fp32 = read_raw(pathResult_CPUFP32+str(i)+"/"+classProbsFile)
    calculated_probs_cpu = read_raw(pathResult_CPUI8+str(i)+"/"+classProbsFile)
    calculated_probs_host = read_raw(pathResult_HOSTI8+str(i)+"/"+classProbsFile)
    calculated_probs_gpu = read_raw(pathResult_GPUI8+str(i)+"/"+classProbsFile)
    calculated_probs_htp = read_raw(pathResult_HTPI8+str(i)+"/"+classProbsFile)
    
    reshaped_probabilities_fp32 = calculated_probs_fp32.reshape(32, 1000)
    max_probabilities_fp32 = np.argmax(reshaped_probabilities_fp32, axis=1)
    
    reshaped_probabilities_cpu = calculated_probs_cpu.reshape(32, 1000)
    max_probabilities_cpu = np.argmax(reshaped_probabilities_cpu, axis=1)
    
    reshaped_probabilities_host = calculated_probs_host.reshape(32, 1000)
    max_probabilities_host = np.argmax(reshaped_probabilities_host, axis=1)
    
    reshaped_probabilities_gpu = calculated_probs_gpu.reshape(32, 1000)
    max_probabilities_gpu = np.argmax(reshaped_probabilities_gpu, axis=1)
    
    reshaped_probabilities_htp = calculated_probs_htp.reshape(32, 1000)
    max_probabilities_htp = np.argmax(reshaped_probabilities_htp, axis=1)
    
    # read the correct labels
    given_labels = read_csv(pathLabel+str(i)+".csv")
    print("=====================================")
    print("Result ",i)
    print("FP32  | CPU   | HOST  | GPU   | HTP")
    
    num_correct_fp32 = 0
    num_correct_cpu = 0
    num_correct_host = 0
    num_correct_gpu = 0
    num_correct_htp = 0
    
    for i in range(len(given_labels)):
        prob_fp32 = max_probabilities_fp32[i]
        prob_cpu = max_probabilities_cpu[i]
        prob_host = max_probabilities_host[i]
        prob_gpu = max_probabilities_gpu[i]
        prob_htp = max_probabilities_htp[i]
        
        label = int(given_labels[i][0])
        
        if prob_fp32 == label:
            num_correct_fp32+=1
        if prob_cpu == label:
            num_correct_cpu+=1
        if prob_host == label:
            num_correct_host+=1
        if prob_gpu == label:
            num_correct_gpu+=1
        if prob_htp == label:
            num_correct_htp+=1
            
        accuracy_fp32 = (num_correct_fp32 / 32) * 100
        accuracy_cpu = (num_correct_cpu / 32) * 100
        accuracy_host = (num_correct_host / 32) * 100
        accuracy_gpu = (num_correct_gpu / 32) * 100
        accuracy_htp = (num_correct_htp / 32) * 100
        
    print("%.2f%%| %.2f%%| %.2f%%| %.2f%%| %.2f%%" % (accuracy_fp32, accuracy_cpu, accuracy_host, accuracy_gpu, accuracy_htp)) 
        
    

    
    
