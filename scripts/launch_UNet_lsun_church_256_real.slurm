#!/bin/bash
###  replace the .py file in the horovodrun call with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err

#SBATCH -J unet_256_real
#SBATCH -o unet_256_real_%j.out
#SBATCH -e unet_256_real_%j.err

#SBATCH --mail-user=seungwook.han@ibm.com
#SBATCH --mail-type=ALL
###  the following parameters will get you two nodes with 4 V100's each
###  note the run time limit is currently only 12 hours, it will be increased
###  next week
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=500g
#SBATCH --time=24:00:00
#SBATCH -p sched_system_all 

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=wmlce-ea
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date


## Horovod execution
###  replace the .py with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err 

python src/train_unet_256_real.py \
--dataset lsun-church_outdoor \
--train_dir $HOME2/data/lsun/ --valid_dir $HOME2/data/lsun/ \
--batch_size 64 --image_size 256 --mask_dim 128 --lr 1e-4 \
--num_epochs 100 --output_dir $HOME2/not-so-biggan-decoder/results/unet_lsun_church_256_real/ \
--project_name unet_lsun_church_256_real \
--save_every 2000 --valid_every 2000 --log_every 100

echo "Run completed at:- "
date
