#!/bin/bash

# command line args
MODEL=$1
DATASET=$2
SPLIT=$3

# if PARTITION is unspecified, throw an error
if [[ -z $PARTITION ]]; then
    echo "PARTITION is unspecified"
    exit 1
fi

# default slurm args
TIME=2-0
CPUS=8
LOG=slurm-logs
# if GPU is unspecified, default to L40S
if [[ -z $GPU ]]; then
    GPU=L40S
fi

# model classes by size
SMALL=(
    "Llama31_8B*"
    "Llama32_1B*"
    "Llama32_3B*"
    "Qwen25_0p5B*"
    "Qwen25_1p5B*"
    "Qwen25_3B*"
    "Qwen25_7B*"
    "Qwen25_14B_*_8K_*"
    "Qwen25_14B_*_16K_*"
    "Qwen25_14B_*_24K_*"
    "Qwen25_14B_*_32K_*"
    "CommandR_7B*"
    "ProLong64K*"
    "ProLong512K*"
)
MEDIUM=(
    "Phi3Mini*"
    "Phi3Small*"
    "Phi3Medium*"
)
LARGE=(
    "Llama33_70B*"
    "Qwen25_14B_*_40K_*"
    "Qwen25_14B_*_48K_*"
    "Qwen25_14B_*_56K_*"
    "Qwen25_14B_*_64K_*"
    "Qwen25_14B_*_72K_*"
    "Qwen25_14B_*_80K_*"
    "Qwen25_14B_128K*"
    "Qwen25_14B_1M*"
    "Qwen25_32B*"
    "Qwen25_72B*"
    "CommandR_32B*"
    "Jamba15Mini*"
)

for pattern in "${SMALL[@]}"; do
    if [[ $MODEL == $pattern ]]; then
        GRES="gpu:$GPU:1"
        MEM=70G
        break
    fi
done
for pattern in "${MEDIUM[@]}"; do
    if [[ $MODEL == $pattern ]]; then
        GRES="gpu:$GPU:2"
        MEM=70G
        break
    fi
done
for pattern in "${LARGE[@]}"; do
    if [[ $MODEL == $pattern ]]; then
        GRES="gpu:$GPU:4"
        MEM=140G
        break
    fi
done
for pattern in "${A100[@]}"; do
    if [[ $MODEL == $pattern ]]; then
        GRES="gpu:A100_80GB:1"
        MEM=70G
        break
    fi
done

ARGS="--time=$TIME"
ARGS="$ARGS --mem=$MEM"
ARGS="$ARGS --cpus-per-task=$CPUS"
ARGS="$ARGS --gres=$GRES"
ARGS="$ARGS --output=$LOG/sum-%j.out"
ARGS="$ARGS --export=ALL,MODEL=$MODEL,DATASET=$DATASET,SPLIT=$SPLIT"

echo "---------------------------------------------------------"
# optional slurm args
if [[ ! -z $DEPD ]]; then
    ARGS="$ARGS --dependency=afterany:$DEPD"
fi
if [[ ! -z $EXCLUDE ]]; then
    ARGS="$ARGS --exclude=$EXCLUDE"
fi
if [[ ! -z $PARTITION ]]; then
    ARGS="$ARGS --partition=$PARTITION"
fi
if [[ ! -z $NODELIST ]]; then
    ARGS="$ARGS --nodelist=$NODELIST"
fi

echo "ARGS:"
echo "$ARGS" | tr ' ' '\n' | sed 's/^/  /'
sbatch --parsable $ARGS \
    bash_scripts/sum.slurm
echo "---------------------------------------------------------"
