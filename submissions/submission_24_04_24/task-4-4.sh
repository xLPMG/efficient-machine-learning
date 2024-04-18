#!/usr/bin/env bash
##
# Example Draco job script.
##
#SBATCH --job-name=mlp_training
#SBATCH --output=mlp_training_%j.out
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

# activate conda environment
module load tools/anaconda3/2021.05
source "$(conda info -a | grep CONDA_ROOT | awk -F ' ' '{print $2}')"/etc/profile.d/conda.sh
conda activate pytorch_x86

python task-4.py