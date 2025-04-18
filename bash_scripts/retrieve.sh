#!/bin/bash

# command line args
MODEL=$1
DATASET=$2
SPLIT=$3

# default slurm args
TIME=2-0
MEM=75G
CPUS=1
LOG=slurm-logs
# if MODEL starts with BM25, GRES is not needed
if [[ $MODEL == BM25* ]]; then
    GRES=""
elif [[ ! -z $GPU ]]; then
    GRES="gpu:$GPU:1"
else
    GRES="gpu:L40S:1"
fi

ARGS="--time=$TIME"
ARGS="$ARGS --mem=$MEM"
ARGS="$ARGS --cpus-per-task=$CPUS"
if [[ ! -z $GRES ]]; then
    ARGS="$ARGS --gres=$GRES"
fi
ARGS="$ARGS --output=$LOG/retrieve-%j.out"
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
    bash_scripts/retrieve.slurm
echo "---------------------------------------------------------"
