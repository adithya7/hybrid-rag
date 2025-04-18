"""Utils for retriever systems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nltk
import tiktoken
from loguru import logger

from models.retriever import BM25Retriever, FlatRetriever

if TYPE_CHECKING:
    from datasets import Dataset

    from configs.datasets import SummDataset
    from configs.retriever import BaseRetrieverConfig


tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count number of tokens, using OpenAI's tokenizer."""
    return len(tiktoken_tokenizer.encode(text))


def segment_docs(
    example: dict,
    doc_key: str,
    max_segment_tokens: int,
) -> dict:
    """
    Truncate document to max_segment_tokens.

    Only truncate at sentence boundaries.
    """
    truncated_docs = []
    for doc in example[doc_key]:
        sents = nltk.sent_tokenize(doc)
        truncated_doc = ""
        for sent in sents:
            if count_tokens(truncated_doc + " " + sent) > max_segment_tokens:
                break
            truncated_doc += " " + sent
        truncated_docs += [truncated_doc]
    example[doc_key] = truncated_docs
    return example


def update_docs(example: dict, idx: int, doc_key: str, docs: list[str]) -> dict:
    """Update examples with retrieved docs."""
    example[doc_key] = docs[idx]
    return example


def preprocess_rag(
    dataset: Dataset,
    dataset_config: SummDataset,
    retriever_config: BaseRetrieverConfig,
    batch_size: int = 200,
) -> Dataset:
    """Preprocess documents for RAG."""
    # setup retriever
    retriever_classes = {
        "Flat": FlatRetriever,
        "BM25": BM25Retriever,
    }
    if retriever_config.retriever_class not in retriever_classes:
        msg = f"retriever class {retriever_config.retriever_class} not supported."
        raise NotImplementedError(msg)

    retriever = retriever_classes[retriever_config.retriever_class](retriever_config)

    # truncate each document to max_segment_tokens
    truncated_dataset = dataset.map(
        segment_docs,
        batched=False,
        desc=f"truncating each doc to {retriever_config.max_segment_tokens} tokens",
        load_from_cache_file=False,
        num_proc=1,
        fn_kwargs={
            "max_segment_tokens": retriever_config.max_segment_tokens,
            "doc_key": dataset_config.doc_key,
        },
    )
    # prepare docs and queries
    docs = truncated_dataset[dataset_config.doc_key]
    queries = dataset[dataset_config.query_key] if dataset_config.query_key else None

    selected_segments = []
    for idx in range(0, len(docs), batch_size):
        logger.info("retrieving docs {}/{}", idx, len(docs))
        batch_selected_segments = retriever.retrieve_docs(
            docs[idx : idx + batch_size],
            queries=queries[idx : idx + batch_size] if queries else None,
            max_retrieved_tokens=retriever_config.max_retrieved_tokens,
        )
        selected_segments += batch_selected_segments

    return dataset.map(
        update_docs,
        with_indices=True,
        batched=False,
        desc="populating retrieved docs",
        load_from_cache_file=False,
        num_proc=1,
        fn_kwargs={"doc_key": dataset_config.doc_key, "docs": selected_segments},
    )
