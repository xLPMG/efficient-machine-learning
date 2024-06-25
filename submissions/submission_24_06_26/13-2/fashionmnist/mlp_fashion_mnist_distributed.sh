#!/usr/bin/env bash
#SBATCH --job-name=mlp_training
#SBATCH --output=out/mlp_training_%j.out
#SBATCH -p short
#SBATCH -N 1-4                   # Number of nodes (1 to 4)
#SBATCH --cpus-per-task=36       # Number of CPUs per MPI process (one full NUMA domain)
#SBATCH --time=01:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=luca-philipp.grumbach@uni-jena.de

echo "Submit host:"
echo $SLURM_SUBMIT_HOST
echo "Submit dir:"
echo $SLURM_SUBMIT_DIR
echo "Nodelist:"
echo $SLURM_JOB_NODELIST

module load anaconda3/2024.02-1
module load mpi/openmpi/4.1.1

conda activate /work/EML/pytorch_env_new

# Different configurations
for nodes in 1 2 4; do
    for mpi_processes in 1 2 4 8; do
        if (( mpi_processes <= nodes * 2 )); then
            OMP_NUM_THREADS=18 mpiexec -n $mpi_processes --bind-to socket --report-bindings python mlp_fashion_mnist_distributed.py > out/mlp_fashion_mnist_distributed_${nodes}nodes_${mpi_processes}procs.out
        fi
    done
done