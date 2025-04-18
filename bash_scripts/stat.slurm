#!/bin/bash

#SBATCH --time 1-0
#SBATCH --mem 30G
#SBATCH --output slurm-logs/stat-dataset-%j.out

DATASET=$1
SPLIT=$2

ARTIFACTS=artifacts
# this is needed for HF dataset loading scripts
export HF_DATA_PATH=artifacts

python src/stat_dataset.py \
    --dataset-config-name $DATASET \
    --split $SPLIT \
    --artifacts-dir $ARTIFACTS
