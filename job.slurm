#!/bin/bash

#SBATCH --job-name=parallel_sampling
#SBATCH --partition="normal"
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mem=16G 
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4 
#SBATCH --exclude=haicn1701
#SBATCH --output=/home/scc/bl7340/res/slurm-%j.out # Change to your output directory
#SBATCH --error=/home/scc/bl7340/res/error-%j.log          # Error log


# Load necessary modules


# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=29500

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PYDIR=/hkfs/work/workspace_haic/scratch/bl7340-parallel_sampling/torch_bayesian # Set path to your python scripts.

# Activate environment
source $PYDIR/myvenv/bin/activate

# Launch distributed training
#srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NTASKS \
#    python -m torch.distributed.run \
#    --nproc_per_node=4 \
#    --nnodes=$SLURM_NNODES \
#    --node_rank=$SLURM_NODEID \
#    --master_addr=$MASTER_ADDR \
#    --master_port=29500 \

srun --ntasks=4 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  echo "Rank $SLURM_PROCID using GPU $CUDA_VISIBLE_DEVICES on node $SLURM_NODEID"
  python $PYDIR/scripts/parallel_sampling/transformer_entsoe.py
'