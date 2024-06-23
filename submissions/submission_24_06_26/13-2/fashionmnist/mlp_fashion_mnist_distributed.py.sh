#!/usr/bin/env bash
##
# Example Draco job script.
##
#SBATCH --job-name=mlp_training
#SBATCH --output=out/mlp_training_%j.out
#SBATCH -p short
#SBATCH -N 1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=luca-philipp.grumbach@uni-jena.de

echo "submit host:"
echo $SLURM_SUBMIT_HOST
echo "submit dir:"
echo $SLURM_SUBMIT_DIR
echo "nodelist:"
echo $SLURM_JOB_NODELIST

module load anaconda3/2024.02-1
module load mpi/openmpi/4.1.1

conda activate /work/EML/pytorch_env
python task-4.py