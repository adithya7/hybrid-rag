#!/bin/bash

echo "============"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "SLURM_CPUS_PER_GPU: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_EXPORT_ENV: $SLURM_EXPORT_ENV"
echo "============"

ARTIFACTS=artifacts
OUTPUT=outputs
# this is needed for HF dataset loading scripts
export HF_DATA_PATH=artifacts

python src/score.py \
    --model-config-name $MODEL \
    --dataset-config-name $DATASET \
    --split $SPLIT \
    --artifacts-dir $ARTIFACTS \
    --output-dir $OUTPUT \
    --tgt-pooling $POOLING \
    --metrics $METRICS
