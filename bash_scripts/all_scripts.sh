#!/bin/bash

# retrieval
DATASET=SummHay; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/retrieve.sh Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT; done;
DATASET=SummHay; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/retrieve.sh Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT; done;

# prepare silver references
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Llama33_70B $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Qwen25_14B_1M $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Qwen25_72B_128K $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Jamba15Mini $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh ProLong512K $DATASET $SPLIT;

# prepare dataset for context length estimation
DATASET=SummHay; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sample.sh Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT; done;
DATASET=SummHay; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sample.sh Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT; done;

bash bash_scripts/estimate.sh

# get context window estimates based on sampled silver data
# most performant and most efficient
# prepare bash scripts for full dataset prediction
for EMB in GTE_1p5B GTE_7B; do
    for SETUP in 5SysPool Qwen25_72B_128K Llama33_70B; do
        python src/get_estimates.py \
            --file-path outputs/SummHay_${SETUP}_Silver_${EMB}_test_scores.csv
    done;
done;

# collect bash scripts needed to run prediction on full dataset into a single script
for EMB in GTE_1p5B GTE_7B; do
    for SETUP in 5SysPool Qwen25_72B_128K Llama33_70B; do
        cat outputs/SummHay_${SETUP}_Silver_${EMB}_test_scores_most_efficient.sh;
    done;
done;

for EMB in GTE_1p5B; do
    for SETUP in 5SysPool Qwen25_72B_128K Llama33_70B; do
        cat outputs/SummHay_${SETUP}_Silver_${EMB}_test_scores_most_performant.sh;
    done;
done;

# run predictions on the full dataset using the above output

# write bash file that needs to be run to get the full dataset scores
# most performant and most efficient estimates
for EMB in GTE_1p5B GTE_7B; do
    for SETUP in 5SysPool Qwen25_72B_128K Llama33_70B; do
        python src/get_estimates.py \
            --file-path outputs/SummHay_${SETUP}_Silver_${EMB}_test_scores.csv \
            --full-dataset SummHay
    done;
done;

# collect scores
# most performant
for EMB in GTE_1p5B GTE_7B; do
    echo "-------------------${EMB}-------------------";
    for SETUP in 5SysPool Qwen25_72B_128K Llama33_70B; do
        echo "-------------------${SETUP}-------------------";
        cat outputs/SummHay_${SETUP}_Silver_${EMB}_test_scores_most_performant_SummHay_test.sh;
        echo "------------------------------------";
    done;
    echo "------------------------------------";
done;
# most efficient
for EMB in GTE_1p5B GTE_7B; do
    echo "-------------------${EMB}-------------------";
    for SETUP in 5SysPool Qwen25_72B_128K Llama33_70B; do
        echo "-------------------${SETUP}-------------------";
        cat outputs/SummHay_${SETUP}_Silver_${EMB}_test_scores_most_efficient_SummHay_test.sh;
        echo "------------------------------------";
    done;
    echo "------------------------------------";
done;
