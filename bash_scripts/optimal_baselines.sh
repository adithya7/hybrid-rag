#!/bin/bash

# full context

# for 0.5B, 1.5B, 3B, we do RAG with 32K context window (their max_position_embeddings);
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_0p5B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_0p5B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
# rest are full context
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_128K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_128K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_32B_128K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B_128K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT

DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT

DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT

# for ProLong64K, we do RAG with 64K context window (its max_position_embeddings)
DATASET=SummHay; SPLIT=test; MODEL=ProLong64K; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=ProLong64K; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
# rest are full context
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT

# collect scores
# for 0.5B, 1.5B, 3B, we do RAG with 32K context window (their max_position_embeddings);
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_0p5B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_0p5B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
# rest are full context
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_128K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_128K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_32B_128K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B_128K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)

DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)

DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)

# for ProLong64K, we do RAG with 64K context window (its max_position_embeddings)
DATASET=SummHay; SPLIT=test; MODEL=ProLong64K; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=ProLong64K; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
# rest are full context
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)

# RULER estimates

# GTE 1.5B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_32B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B_128K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT

DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT

DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT
# GTE 7B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_32B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B_128K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT

DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT

DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_7B $DATASET $SPLIT
---
# collect scores
# GTE 1.5B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_32B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B_128K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)

# GTE 7B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_32B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B_128K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)

# HELMET estimates (based on summarization task avg.)

DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context
DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context
DATASET=SummHay; SPLIT=test; MODEL=Jamba15Mini; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context

# with GTE 1.5B, GTE 7B
for EMB in GTE_1p5B GTE_7B; do
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT

    DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT

    DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
done;

# collect scores
# GTE 1.5B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)

# GTE 7B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)

# HELMET estimates (based on LongQA task avg.)

DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; bash bash_scripts/sum.sh $MODEL $DATASET $SPLIT # full 128k context

# with GTE 1.5B, GTE 7B
for EMB in GTE_1p5B GTE_7B; do
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=16K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=16K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT

    DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; CONTEXT=32K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT

    DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
    DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; CONTEXT=64K; bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT
done;

# collect scores
# GTE 1.5B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=16K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=16K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)

# GTE 7B
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_1p5B; CONTEXT=16K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_3B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B; CONTEXT=16K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_7B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_14B_1M; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Qwen25_72B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_1B; CONTEXT=32K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama32_3B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama31_8B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Llama33_70B; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Mini; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Small; CONTEXT=64K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_7B_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=Phi3Medium; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
DATASET=SummHay; SPLIT=test; MODEL=ProLong512K; awk '{print $1","$6","$7}' <(tail -n1 outputs/${DATASET}_${MODEL}_${SPLIT}_scores.txt)
