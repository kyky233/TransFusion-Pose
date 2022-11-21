#!/bin/bash

#SBATCH -J tp_mvhw_train
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:3
#SBATCH -o /mntnfs/med_data5/junhaoran/TransFusion-Pose/slurm_logs/train_mvhw_%j.out
#SBATCH -e /mntnfs/med_data5/junhaoran/TransFusion-Pose/slurm_logs/train_mvhw_%j.out
#SBATCH --mail-type=ALL  # BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yandq2020@mail.sustech.edu.cn

# export MASTER_PORT=$((12000 + $RANDOM % 2000))
set -x
CONFIG=experiments-local/mvhw/transdpose/train_mvhw_joints_15_256x256_neighbor_cams.yaml

# PYTHONPATH="$(dirname ./scripts/train_mvhw_15.sh)/..":$PYTHONPATH \
which python

python -m torch.distributed.launch --nproc_per_node=1 --use_env run/pose2d/train.py --cfg $CONFIG
