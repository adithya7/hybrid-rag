"""Retriever script."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fire
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from loguru import logger

from config_utils import init_dataset_config, init_retriever_config
from retriever_utils import preprocess_rag

if TYPE_CHECKING:
    from configs.datasets import SummDataset

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | "
        "<c>{name}</c>:<c>{function}</c>:<c>{line}</c> | {message}"
    ),
)


def load_examples(config: SummDataset, split: str, debug: bool = False) -> Dataset:
    """Load dataset (HF format)."""
    if getattr(config, "load_from_disk", False):
        # for prefilterd datasets
        logger.info("loading dataset: {} | {}", config.path, split)
        dataset = load_from_disk(config.path)
        dataset = dataset[split]
    else:
        logger.info("loading dataset: {} | {} | {}", config.path, config.name, split)
        dataset = load_dataset(
            path=config.path,
            name=config.name,
            split=split,
            trust_remote_code=True,
        )
    if debug:
        logger.warning("debug mode, using only 100 examples")
        dataset = dataset.select(range(100))

    # flatten docs in each example, if list of lists
    if isinstance(dataset[0][config.doc_key][0], list):
        logger.info("found list of lists in docs, flattening")
        dataset = dataset.map(
            lambda example: {
                **example,
                config.doc_key: [
                    doc for docs in example[config.doc_key] for doc in docs
                ],
            },
            batched=False,
            desc="flattening docs",
            load_from_cache_file=False,
            num_proc=1,
        )

    # add query prompt (if any)
    if hasattr(config, "query_prompt"):
        dataset = dataset.map(
            lambda example: {
                **example,
                "query": config.query_prompt.format(query=example[config.query_key]),
            },
            batched=False,
            desc="adding query prompt",
            load_from_cache_file=False,
            num_proc=1,
        )

    return dataset


def main(
    retriever_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
    output_dir: str,
) -> None:
    """Retrieve documents."""
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)

    # setup retriever config
    retriever_config = init_retriever_config(retriever_config_name, artifacts_dir)
    retriever_config.output_dir = (
        output_dir / f"retriever_{dataset_config_name}_{retriever_config_name}"
    )
    # setup dataset config
    dataset_config = init_dataset_config(dataset_config_name, artifacts_dir)
    # load dataset
    dataset = load_examples(dataset_config, split)
    # select input segments for RAG
    dataset = preprocess_rag(
        dataset=dataset,
        dataset_config=dataset_config,
        retriever_config=retriever_config,
    )
    dataset = DatasetDict({split: dataset})
    logger.info("saving retrieved dataset to {}", retriever_config.output_dir)
    dataset.save_to_disk(retriever_config.output_dir)


if __name__ == "__main__":
    fire.Fire(main)
