#!/bin/bash

#SBATCH --time 0-10
#SBATCH --mem 50G
#SBATCH --partition general
#SBATCH --output=slurm-logs/download-%j.out

export HF_HOME=huggingface

HF_TOKEN=$(cat tokens/hf_access_token.txt)
huggingface-cli login --token $HF_TOKEN

LOCAL_DIR=artifacts/huggingface

for MODEL in 0.5B 1.5B 3B 7B 14B 32B 72B; do
    python src/download.py \
        --repo-id Qwen/Qwen2.5-${MODEL}-Instruct \
        --local-dir $LOCAL_DIR \
        --repo-type model
done

python src/download.py \
    --repo-id meta-llama/Llama-3.1-8B-Instruct \
    --local-dir $LOCAL_DIR \
    --repo-type model

for MODEL in 1.5B 7B; do
    python src/download.py \
        --repo-id Alibaba-NLP/gte-Qwen2-${MODEL}-instruct \
        --local-dir $LOCAL_DIR \
        --repo-type model
done

for MODEL in Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct Llama-3.3-70B-Instruct; do
    python src/download.py \
        --repo-id meta-llama/$MODEL \
        --local-dir $LOCAL_DIR \
        --repo-type model
done

for MODEL in mini small medium; do
    python src/download.py \
        --repo-id microsoft/Phi-3-${MODEL}-128k-instruct \
        --local-dir $LOCAL_DIR \
        --repo-type model
done

for MODEL in 64k 512k; do
    python src/download.py \
        --repo-id princeton-nlp/Llama-3-8B-ProLong-${MODEL}-Instruct \
        --local-dir $LOCAL_DIR \
        --repo-type model
done

for MODEL in 7B 14B; do
    python src/download.py \
        --repo-id Qwen/Qwen2.5-${MODEL}-Instruct-1M \
        --local-dir $LOCAL_DIR \
        --repo-type model
done
