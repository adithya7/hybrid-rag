"""
Sample examples for context window estimation.

Load examples from retrieved datasets.
Extract summaries from system predictions and compile a dataset.
(optionally) use MBR decoding to identify the best system summaries.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import numpy as np
from autoacu import A3CU
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from loguru import logger
from tqdm import tqdm

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

SYSTEMS = [
    "Llama33_70B",
    "Qwen25_14B_1M",
    "Qwen25_72B_128K",
    "Jamba15Mini",
    "ProLong512K",
]


def load_examples(config: SummDataset, split: str) -> Dataset:
    """Load dataset (HF format)."""
    logger.info("loading dataset: {} | {} | {}", config.path, config.name, split)
    return load_dataset(
        path=config.path,
        name=config.name,
        split=split,
        trust_remote_code=True,
    )


def load_predictions(path: Path) -> list[str]:
    """Load predictions from file."""
    preds = []
    logger.info("loading predictions from: {}", path)
    with path.open("r") as rf:
        for line in rf:
            ex_pred = json.loads(line)["pred"]
            if isinstance(ex_pred, str):
                ex_pred = [ex_pred]
            preds += [ex_pred]
    return preds


def mbr_inference(
    preds: list[list[str]], model_pt: Path, k: int = 3
) -> list[list[str]]:
    """Perform MBR decoding."""
    a3cu = A3CU(model_pt=model_pt)
    selected_preds = []
    counter = defaultdict(int)
    for idx in tqdm(range(len(preds))):
        scores = []
        for idy in range(len(preds[idx])):
            _, _, f1 = a3cu.score(
                references=preds[idx],
                candidates=[preds[idx][idy]] * len(preds[idx]),
                batch_size=16,
                verbose=False,
            )
            scores += [np.mean(f1)]
        # select top-k predictions
        selected_preds += [[preds[idx][idy] for idy in np.argsort(scores)[::-1][:k]]]
        for idy in np.argsort(scores)[::-1][:k]:
            # for each SYSTEM, we have three summaries in preds[idx]
            counter[SYSTEMS[idy // 3]] += 1
    logger.info("top-k system counter: {}", counter)
    return selected_preds


def main(  # noqa: PLR0913
    retriever_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
    output_dir: str,
    sampling_ratio: float = 0.25,
    use_mbr: bool = False,
    reference_model: str | None = None,
) -> None:
    """Sample examples and prepare dataset for context window estimation."""
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)

    # load dataset
    dataset_path = (
        output_dir / f"retriever_{dataset_config_name}_{retriever_config_name}"
    )
    logger.info("loading dataset from: {}", dataset_path)
    dataset = load_from_disk(dataset_path)
    dataset = dataset[split]
    logger.info("len(dataset): {}", len(dataset))

    # load predictions
    reference_systems = [reference_model] if reference_model else SYSTEMS
    logger.info("reference systems: {}", ",".join(reference_systems))
    pred_paths = [
        output_dir / f"{dataset_config_name}_{model_config_name}_{split}.jsonl"
        for model_config_name in reference_systems
    ]
    sys_preds = [load_predictions(pred_path) for pred_path in pred_paths]
    sys_preds = [list(chain.from_iterable(preds)) for preds in zip(*sys_preds)]

    if use_mbr:
        # MBR decoding
        # we use A3CU/F1 score as the utility metric
        logger.info("performing MBR decoding")
        a3cu_model_pt = artifacts_dir / "huggingface/model/Yale-LILY/a3cu"
        sys_preds = mbr_inference(sys_preds, a3cu_model_pt)

    # replace gold summaries with system predictions
    dataset = dataset.map(
        lambda example, idx: {
            **example,
            "summary": list(sys_preds[idx]),
        },
        with_indices=True,
        batched=False,
        desc="replacing gold summary with sys preds",
        load_from_cache_file=False,
        num_proc=1,
    )
    n_samples = int(sampling_ratio * len(dataset))
    logger.info("sampling {}/{} examples", n_samples, len(dataset))
    # select n_samples random examples
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    dataset = DatasetDict({split: dataset})
    # save dataset
    output_path = (
        f"retriever_{dataset_config_name}_{reference_model}_Silver_{retriever_config_name}"
        if reference_model
        else f"retriever_{dataset_config_name}_5SysPool_Silver_{retriever_config_name}"
    )
    output_path = output_dir / output_path
    logger.info("saving dataset to: {}", output_path)
    dataset.save_to_disk(output_path)


if __name__ == "__main__":
    fire.Fire(main)
