#!/bin/bash

# command line args
RETRIEVER=$1
DATASET=$2
SPLIT=$3

# default slurm args
TIME=0-2
MEM=20G
GRES="gpu:L40S:1"
CPUS=1
LOG=slurm-logs

ARGS="--time=$TIME"
ARGS="$ARGS --mem=$MEM"
ARGS="$ARGS --cpus-per-task=$CPUS"
ARGS="$ARGS --gres=$GRES"
ARGS="$ARGS --output=$LOG/sample-%j.out"
ARGS="$ARGS --export=ALL,RETRIEVER=$RETRIEVER,DATASET=$DATASET,SPLIT=$SPLIT"

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
if [[ ! -z $NODES ]]; then
    ARGS="$ARGS --nodelist=$NODES"
fi

echo "ARGS:"
echo "$ARGS" | tr ' ' '\n' | sed 's/^/  /'
sbatch --parsable $ARGS \
    bash_scripts/sample.slurm
echo "---------------------------------------------------------"
