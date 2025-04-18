"""Summarization script."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import nltk
from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger
from transformers import AutoTokenizer

from config_utils import init_dataset_config, init_model_config
from data_utils import prepare_queries, truncate_dataset
from llm_api_utils import init_api, predict_api
from vllm_utils import init_vllm, predict_vllm

if TYPE_CHECKING:
    from configs.datasets import SummDataset
    from configs.models import BaseModel

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

nltk.download("punkt_tab", quiet=True)


def predict(  # noqa: PLR0913
    model_config: BaseModel,
    dataset_config: SummDataset,
    docs: list[str],
    queries: list[str],
    tokenizer: AutoTokenizer,
    num_gpus: int,
) -> list[str]:
    """Predict final summaries, either using API or vLLM."""
    if hasattr(model_config, "api"):
        # API-based model inference
        logger.info("using API-based inference...")
        client = init_api(model_config)
        return predict_api(
            client=client,
            docs=docs,
            queries=queries,
            model_config=model_config,
            dataset_config=dataset_config,
        )
    # local inference using vLLM
    # initialize model (and tokenizer)
    logger.info("using vLLM for inference...")
    model = init_vllm(model_config, num_gpus)
    return predict_vllm(
        model=model,
        docs=docs,
        queries=queries,
        tokenizer=tokenizer,
        model_config=model_config,
        dataset_config=dataset_config,
    )


def load_examples(config: SummDataset, split: str, debug: bool = False) -> Dataset:
    """Load dataset (HF format)."""
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

    # NOTE: we don't need to flatten docs here, this is handled in input truncation
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


def main(  # noqa: PLR0913
    model_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
    output_dir: str,
    num_gpus: int | None = None,
) -> None:
    """Get summary predictions."""
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)
    dataset_config = init_dataset_config(dataset_config_name, artifacts_dir)
    model_config = init_model_config(
        model_config_name, dataset_config_name, split, artifacts_dir, output_dir
    )
    if hasattr(model_config, "retriever"):
        # load preprocessed data from the disk
        # already truncated to account for summary tokens
        # query prompt is already added (if any)
        dataset_path = (
            output_dir / f"retriever_{dataset_config_name}_{model_config.retriever}"
        )
        logger.info("loading dataset: {}", dataset_path)
        dataset = load_from_disk(dataset_path)
        dataset = dataset[split]
    else:
        dataset = load_examples(dataset_config, split)
    # setup tokenizer, for both truncation and token_id generation during prediction
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name_or_path,
        trust_remote_code=getattr(model_config, "trust_remote_code", False),
    )
    # set max input length
    dataset_config.max_summary_tokens = round(
        dataset_config.max_summary_words * model_config.word2token_ratio
    )
    max_inp_length = model_config.max_length - dataset_config.max_summary_tokens - 100
    dataset = truncate_dataset(
        dataset=dataset,
        doc_key=dataset_config.doc_key,
        max_length=max_inp_length,
        tokenizer=tokenizer,
        min_doc_tokens=getattr(dataset_config, "min_doc_tokens", None),
    )
    docs = dataset[dataset_config.doc_key]
    # if any item in docs is not a list of string, raise exception
    if any(not isinstance(item, list) for item in docs):
        msg = "docs must be a list of list of strings"
        raise ValueError(msg)
    concatenated_docs = ["\n\n".join(item) for item in docs]
    # summary predictions
    queries = prepare_queries(dataset, dataset_config)
    outputs = predict(
        model_config=model_config,
        dataset_config=dataset_config,
        docs=concatenated_docs,
        queries=queries,
        tokenizer=tokenizer,
        num_gpus=num_gpus,
    )
    # save predictions
    logger.info("saving predictions to {}", model_config.pred_path)
    with model_config.pred_path.open("w") as wf:
        for idx in range(len(outputs)):
            question = dataset[idx].get(
                dataset_config.query_key, dataset_config.default_query
            )
            wf.write(
                json.dumps(
                    {
                        "src": dataset[idx][dataset_config.doc_key],
                        "docs": docs[idx],
                        "pred": outputs[idx],
                        "question": question,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    fire.Fire(main)
