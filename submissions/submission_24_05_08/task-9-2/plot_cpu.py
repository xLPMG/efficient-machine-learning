# PLOT RESULT
output_file = "results.csv"

import matplotlib.pyplot as plt 
import csv 
  
cpuX = [] 
cpuY = [] 
cpuDtype = [] 
  
with open(output_file,'r') as csvfile: 
    plots = csv.reader(csvfile, delimiter = ',') 
      
    for row in plots: 
        if row[0] == 'cpu':
            cpuX.append(float(row[1])) # size
            cpuY.append(float(row[5])) # gflops
            cpuDtype.append(row[2].split(".")[1])    # dtype
            
fig, ax = plt.subplots()

for dtype in set(cpuDtype):
    indices = [i for i, x in enumerate(cpuDtype) if x == dtype]
    plt.plot([cpuX[i] for i in indices], [cpuY[i] for i in indices], label=dtype, linestyle='-', marker='o')

ax.set_xscale('log', base=2)

plt.title("GEMM benchmark results: CPU")
plt.xlabel("Matrix size")
plt.ylabel("GFLOP/s")

# Shrink current axis by 20% and place legend next to it
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.text(0.01, 0.01, "Luca-Philipp Grumbach")
plt.show() 
plt.savefig('visualizations/gemm_benchmark_cpu.png')