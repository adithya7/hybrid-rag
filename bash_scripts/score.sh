#!/bin/bash

# command line args
MODEL=$1
DATASET=$2
SPLIT=$3
POOLING=$4

METRICS=a3cu

# default slurm args
TIME=0-2
MEM=15G
GRES="gpu:1"
CPUS=4
LOG=slurm-logs

ARGS="--time=$TIME"
ARGS="$ARGS --mem=$MEM"
ARGS="$ARGS --cpus-per-task=$CPUS"
ARGS="$ARGS --gres=$GRES"
ARGS="$ARGS --output=$LOG/score-%j.out"
ARGS="$ARGS --export=ALL,MODEL=$MODEL,DATASET=$DATASET,SPLIT=$SPLIT,POOLING=$POOLING,METRICS=$METRICS"

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
    bash_scripts/score.slurm
echo "---------------------------------------------------------"
