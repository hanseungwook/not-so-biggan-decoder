#!/bin/bash
#SBATCH -J in_hier
#SBATCH -o in_hier_%j.out
#SBATCH -e in_hier_%j.err

#SBATCH --mail-user=swhan@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

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
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NOD


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
horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python \
      $HOME2/not-so-biggan-decoder/src/train_vgg19_hf.py \
      --train-dir /data/ImageNet/ILSVRC2012/train/ \
      --val-dir /data/ImageNet/ILSVRC2012/val/ \
      --hf \
      --fp16-allreduce \
      --batch-size=64 \
      --wandb-entity sswwhan \
      --wandb_project_name vgg19-hf \
      --wandb-api-key $HOME2/settings/wandb_api_key.txt \
      --num-wt-levels 3 \
      --epochs 120 \
      --resume-from-epoch \
      --model-load-path . \
      --save-models \
      --save-path $HOME2/not-so-biggan-decoder/results

echo "Run completed at:- "
date
