# Hybrid RAG and Long-context

Code for our preprint, [Estimating Optimal Context Length for Hybrid Retrieval-augmented Multi-document Summarization](https://arxiv.org/abs/2504.12972).

## Setup

See [requirements.txt](requirements.txt).

## Data

We provide the HuggingFace dataset loading script in [artifacts/misc/summhay/](artifacts/misc/summhay/). Before loading the dataset, use the original release to download the source data (see [README](artifacts/misc/summhay/README.md)).

## Experiments

See [src/configs/](src/configs/) for the full list of retrievers, summarizers and datasets used in our experiments.

### Downloading models

Download retriever and embedding models from HuggingFace using [bash_scripts/download.slurm](bash_scripts/download.slurm).

### Baselines

Script to run baselines is provided at [bash_scripts/optimal_baselines.sh](bash_scripts/optimal_baselines.sh). This includes full-context and RAG based on RULER and HELMET task-based estimates. For HELMET, we use two task-based estimates, LongQA and summarization. We use two retrievers, GTE-Qwen-1.5B and GTE-Qwen-7B.

### Proposed

To get optimal estimates for retrieval context length, we use a randomly sampled subset of the evaluation data. We replace the gold summaries in the HuggingFace dataset with silver candidates from our LLM panel (see [src/sample_examples.py](src/sample_examples.py)).

For the full experimental setup, see [bash_scripts/all_scripts.sh](bash_scripts/all_scripts.sh). This includes the following steps,

+ sample a subset of the evaluation dataset for context length estimation
+ retrieval using GTE embeddings
+ collect silver references using a LLM panel
+ estimate optimal context length by evaluating systems (retriever + summarizer) on the randomly sampled subset ([bash_scripts/estimate.sh](bash_scripts/estimate.sh))
+ run prediction on the full evaluation dataset using the optimal context length

Below, we provide the outline for one experimental configuration (Qwen 2.5 7B summarizer with GTE 1.5B retriever).

```bash
# NOTE: in our experiments, we compare against full context and baseline RAG setups
# therefore, we first run retrieval and get silver references on the full dataset before sampling a subset.
# however, this order can be reversed by first sampling a subset before running retrieval and getting silver references.

# first, retrieval using GTE 1.5B embeddings
DATASET=SummHay; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/retrieve.sh Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT; done;

# now we prepare silver references using our LLM panel
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Llama33_70B $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Qwen25_14B_1M $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Qwen25_72B_128K $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh Jamba15Mini $DATASET $SPLIT;
DATASET=SummHay; SPLIT=test; bash bash_scripts/sum.sh ProLong512K $DATASET $SPLIT;

# now, we sample a subset of the evaluation dataset for context length estimation
# for each retriever context length, we generate a new dataset (SummHay_5SysPool_Silver) by replacing the gold summaries with the silver candidates from our LLM panel
DATASET=SummHay; SPLIT=test; for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sample.sh Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT; done;

# prediction on the sampled subset
# for this illustration, we use Qwen25_7B as the summarizer
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_7B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/sum.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT; done; done;
# score against the silver references
# we use max pooling over target summaries
DATASET=SummHay_5SysPool_Silver; SPLIT=test; POOLING=max; for MODEL in Qwen25_7B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do bash bash_scripts/score.sh ${MODEL}_Flat_${CONTEXT}_GTE_1p5B $DATASET $SPLIT $POOLING; done; done;

# collect scores
FILEPATH=outputs/SummHay_5SysPool_Silver_${EMB}_test_scores.csv
head -n1 outputs/SummHay_5SysPool_Silver_Qwen25_7B_Flat_8K_GTE_1p5B_test_scores.txt | sed 's/ \+/,/g' > $FILEPATH
DATASET=SummHay_5SysPool_Silver; SPLIT=test; for MODEL in Qwen25_7B; do for CONTEXT in 8K 16K 24K 32K 40K 48K 56K 64K 72K 80K; do sed 's/ \+/,/g' <(tail -n1 outputs/${DATASET}_${MODEL}_Flat_${CONTEXT}_GTE_1p5B_${SPLIT}_scores.txt) >> $FILEPATH; done; done;

# get estimates
python src/get_estimates.py \
    --file-path outputs/SummHay_5SysPool_Silver_GTE_1p5B_test_scores.csv

# this gives the script to run on the full evaluation dataset
cat outputs/SummHay_5SysPool_Silver_GTE_1p5B_test_scores_most_efficient.sh;
```

## License

This project is licensed under the MIT License. See the LICENSE file.

## Reference

If you find this work useful, please consider citing our paper.


```bibtex
@misc{pratapa-mitamura-2025-estimating,
    title={Estimating Optimal Context Length for Hybrid Retrieval-augmented Multi-document Summarization},
    author={Adithya Pratapa and Teruko Mitamura},
    year={2025},
    eprint={2504.12972},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2504.12972},
}
```
