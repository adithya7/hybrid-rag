"""Get aggregate statistics for a dataset."""

import sys
from pathlib import Path

import fire
import nltk
import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm

from config_utils import get_dataset_config

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)

nltk.download("punkt_tab")
tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")


def log_stats(values: list) -> None:
    """Log aggregate stats."""
    logger.info(
        "mean: {}, min: {}, max: {}", np.mean(values), np.min(values), np.max(values)
    )
    for percentile in [70, 80, 90, 95]:
        logger.info(
            "{}th percentile: {}", percentile, np.percentile(values, percentile)
        )
    logger.info("-" * 50)


def stat_dataset(
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
) -> None:
    """Get dataset statistics."""
    artifacts_dir = Path(artifacts_dir)
    logger.info("loading dataset {}, split {}", dataset_config_name, split)
    config = get_dataset_config(dataset_config_name)
    config.path = str(artifacts_dir / config.path)
    # use all splits for stats
    if dataset_config_name == "BackgroundFiltered":
        dataset = load_from_disk(config.path)
        dataset = dataset[split]
    else:
        dataset = load_dataset(
            path=config.path,
            name=config.name,
            split=split,
            trust_remote_code=True,
        )

    # for multi-reference summary, use mean summary length per example
    summary_len = [
        len(nltk.word_tokenize(s))
        if isinstance(s, str)
        else np.mean([len(nltk.word_tokenize(_item)) for _item in s])
        for s in dataset[config.summary_key]
    ]
    logger.info("summary length stats (nltk tokens) ----->")
    log_stats(summary_len)

    # tiktoken tokenizer
    summary_len = [
        len(tiktoken_tokenizer.encode(s))
        if isinstance(s, str)
        else np.mean([len(tiktoken_tokenizer.encode(_item)) for _item in s])
        for s in dataset[config.summary_key]
    ]
    logger.info("summary length stats (openai tokens) ----->")
    log_stats(summary_len)

    all_docs = dataset[config.doc_key]
    # flatten docs to have a single list of doc strings per summary
    if isinstance(all_docs[0][0], list):
        # flatten list of list of documents
        all_docs = [[item for sublist in doc for item in sublist] for doc in all_docs]

    if not isinstance(all_docs[0][0], str):
        msg = "Document is not a string"
        raise TypeError(msg)

    # plot number of documents per summary
    num_docs = [len(doc) for doc in all_docs]
    logger.info("number of documents per summary stats ----->")
    log_stats(num_docs)

    # dedup docs
    all_docs = [list(set(doc)) for doc in all_docs]
    num_docs = [len(doc) for doc in all_docs]
    logger.info("number of unique documents per summary stats ----->")
    log_stats(num_docs)

    # plot document length distribution
    doc_len = []
    for topic_docs in tqdm(all_docs, colour="green"):
        for doc in topic_docs:
            doc_len += [len(nltk.word_tokenize(doc))]
    logger.info("(unique) document length stats (nltk tokens) ----->")
    log_stats(doc_len)

    # tiktoken tokenizer
    doc_len = []
    for topic_docs in tqdm(all_docs, colour="green"):
        for doc in topic_docs:
            doc_len += [len(tiktoken_tokenizer.encode(doc))]
    logger.info("(unique) document length stats (openai tokens) ----->")
    log_stats(doc_len)


if __name__ == "__main__":
    fire.Fire(stat_dataset)
