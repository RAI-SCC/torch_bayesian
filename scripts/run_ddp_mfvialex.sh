#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --account='hk-project-epais'
#SBATCH --output='tuned_ddp_mfvialex.out'
#SBATCH --job-name='ddp_mfvialex_test'

module purge # Unload all models.
module load ddp_mfvialex

source ../.venv/bin/activate
which python

export PYTHONPATH=$PYTHONPATH:~/vi_tests/vi

# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=13349

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun python -u DDP_MFVIAlexNet.py
