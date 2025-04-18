#!/bin/bash

# EMB=GTE_1p5B
EMB=GTE_7B

# System pooling (Qwen)
# prediction
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_7B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_7B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_14B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_14B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_32B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_72B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
# scoring
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_7B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_7B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_14B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_14B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_32B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_72B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

# System pooling (Llama)
# prediction
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Llama32_1B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Llama32_3B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Llama31_8B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Llama33_70B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done; done;
# scoring
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Llama32_1B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Llama32_3B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Llama31_8B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Llama33_70B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

# System pooling (Phi3)
# prediction
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh Phi3Mini_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh Phi3Small_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh Phi3Medium_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done;
# scoring
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh Phi3Mini_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh Phi3Small_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh Phi3Medium_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done;

# System pooling (ProLong)
# prediction
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do bash bash_scripts/sum.sh ProLong64K_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ProLong512K_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT; done;
# scoring
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do bash bash_scripts/score.sh ProLong64K_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ProLong512K_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done;

# Qwen 72B based silver references
# prediction is not needed, since the inputs are the same as SummHay_5SysPool_Silver, just reuse predictions
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Qwen25_7B Qwen25_7B_1M Qwen25_14B Qwen25_14B_1M Qwen25_32B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
# scoring
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_7B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_7B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_14B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_14B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_32B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
# Llama 70B based silver references
# prediction
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Llama32_1B Llama32_3B Llama31_8B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
# scoring
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Llama32_1B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Llama32_3B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Llama31_8B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

# cross-model estimates

# estimates for Llama using Qwen silver summaries
# prediction
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Llama32_1B Llama32_3B Llama31_8B Llama33_70B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Phi3Mini Phi3Small Phi3Medium; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in ProLong64K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in ProLong512K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;

# scoring
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Llama32_1B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Llama32_3B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Llama31_8B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Llama33_70B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Phi3Mini; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Phi3Small; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in Phi3Medium; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in ProLong64K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; POOLING=max; for MODEL in ProLong512K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

# estimates for Qwen using Llama silver summaries

# prediction
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Qwen25_7B Qwen25_7B_1M Qwen25_14B Qwen25_14B_1M Qwen25_32B Qwen25_72B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Phi3Mini Phi3Small Phi3Medium; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in ProLong64K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in ProLong512K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do cp SummHay_5SysPool_Silver_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl ${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}.jsonl; done; done;

# scoring
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_7B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_7B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_14B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_14B_1M; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_32B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_72B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Phi3Mini; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Phi3Small; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in Phi3Medium; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in ProLong64K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; POOLING=max; for MODEL in ProLong512K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_${EMB} $DATASET $SPLIT $POOLING; done; done;

# collect scores
FILEPATH=outputs/SummHay_5SysPool_Silver_${EMB}_test_scores.csv
head -n1 outputs/SummHay_5SysPool_Silver_Qwen25_0p5B_Flat_8K_${EMB}_test_scores.txt | sed 's/ \+/,/g' > $FILEPATH

DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_7B Qwen25_7B_1M Qwen25_14B Qwen25_14B_1M Qwen25_32B Qwen25_72B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Llama32_1B Llama32_3B Llama31_8B Llama33_70B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Phi3Mini Phi3Small Phi3Medium; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in ProLong64K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in ProLong512K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;

FILEPATH=outputs/SummHay_Qwen25_72B_128K_Silver_${EMB}_test_scores.csv
head -n1 outputs/SummHay_Qwen25_72B_128K_Silver_Qwen25_0p5B_Flat_8K_${EMB}_test_scores.txt | sed 's/ \+/,/g' > $FILEPATH

DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Qwen25_7B Qwen25_7B_1M Qwen25_14B Qwen25_14B_1M Qwen25_32B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Llama32_1B Llama32_3B Llama31_8B Llama33_70B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in Phi3Mini Phi3Small Phi3Medium; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in ProLong64K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Qwen25_72B_128K_Silver; SPLIT=test; for MODEL in ProLong512K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;

FILEPATH=outputs/SummHay_Llama33_70B_Silver_${EMB}_test_scores.csv
head -n1 outputs/SummHay_Llama33_70B_Silver_Llama32_1B_Flat_8K_${EMB}_test_scores.txt | sed 's/ \+/,/g' > $FILEPATH

DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Qwen25_0p5B Qwen25_1p5B Qwen25_3B; do for CONTEXT in 8K 16K 24K 32K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Qwen25_7B Qwen25_7B_1M Qwen25_14B Qwen25_14B_1M Qwen25_32B Qwen25_72B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Llama32_1B Llama32_3B Llama31_8B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in Phi3Mini Phi3Small Phi3Medium; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in ProLong64K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
DATASET=SummHay_Llama33_70B_Silver; SPLIT=test; for MODEL in ProLong512K; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_${EMB}_${SPLIT}_scores.txt) >> $FILEPATH; done; done;
