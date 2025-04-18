"""Utils."""

from __future__ import annotations

import inspect
from pprint import pformat
from typing import TYPE_CHECKING

from loguru import logger

import configs.datasets as custom_datasets
import configs.models as custom_models
import configs.retriever as custom_retrievers

if TYPE_CHECKING:
    from pathlib import Path


def get_model_config(model_name: str) -> custom_models.BaseModel:
    """Get model config."""
    model_configs = dict(inspect.getmembers(custom_models))
    return model_configs[model_name]()


def get_dataset_config(dataset_name: str) -> custom_datasets.SummDataset:
    """Get dataset config."""
    dataset_configs = dict(inspect.getmembers(custom_datasets))
    return dataset_configs[dataset_name]()


def get_retriever_config(retriever_name: str) -> custom_retrievers.BaseRetrieverConfig:
    """Get retriever config."""
    retriever_configs = dict(inspect.getmembers(custom_retrievers))
    return retriever_configs[retriever_name]()


def init_dataset_config(
    dataset_config_name: str, artifacts_dir: Path
) -> custom_datasets.SummDataset:
    """Load and init dataset config object."""
    dataset_config = get_dataset_config(dataset_config_name)
    # setup dataset path
    if getattr(dataset_config, "path", None) is not None:
        dataset_config.path = str(artifacts_dir / dataset_config.path)
    logger.info(f"dataset config: {pformat(dataset_config)}")
    return dataset_config


def init_model_config(
    model_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: Path,
    output_dir: Path,
) -> custom_models.BaseModel:
    """Load and init model config object."""
    # load model and dataset configs
    model_config = get_model_config(model_config_name)
    # setup model path
    model_config.model_name_or_path = str(
        artifacts_dir / "huggingface/model" / model_config.model_name_or_path
    )
    # setup tokenizer path
    if not hasattr(model_config, "tokenizer_name_or_path"):
        model_config.tokenizer_name_or_path = model_config.model_name_or_path
    else:
        msg = "found tokenizer_name_or_path"
        raise NotImplementedError(msg)
    # setup prediction path
    model_config.output_dir = output_dir
    model_config.output_dir.mkdir(parents=True, exist_ok=True)
    model_config.pred_path = (
        model_config.output_dir
        / f"{dataset_config_name}_{model_config_name}_{split}.jsonl"
    )
    # setup log path
    model_config.log_path = (
        output_dir / "logs" / f"{model_config_name}_{dataset_config_name}_{split}"
    )
    model_config.log_path.mkdir(parents=True, exist_ok=True)
    # add logger
    logger.add(
        str(model_config.log_path) + "/" + "{time}.log",
        format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
    )
    logger.info(f"model config: {pformat(model_config)}")

    return model_config


def init_retriever_config(
    config_name: str,
    artifacts_dir: Path,
) -> custom_retrievers.BaseRetrieverConfig:
    """Load and init retriever config object."""
    retriever_config = get_retriever_config(config_name)
    # setup retriver model path(s), typically the embedding model
    if hasattr(retriever_config, "embedder"):
        retriever_config.embedder = str(
            artifacts_dir / "huggingface/model" / retriever_config.embedder
        )
    if hasattr(retriever_config, "a3cu_model"):
        retriever_config.a3cu_model = str(
            artifacts_dir / "huggingface/model" / retriever_config.a3cu_model
        )
    # log config
    logger.info(f"retriever config: {pformat(retriever_config)}")
    return retriever_config
