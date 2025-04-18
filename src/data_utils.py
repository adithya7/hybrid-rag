"""Data related utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nltk
from loguru import logger

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import AutoTokenizer

    from configs.datasets import SummDataset


def truncate_helper(
    docs: list[str],
    max_inp_tokens: int,
    tokenizer: AutoTokenizer,
    min_doc_tokens: int | None = None,
) -> list[str]:
    """
    Truncate documents within a collection to `max_inp_tokens`.

    Longest documents are truncated first.
    All documents in the collection are related to each other.

    If min_doc_tokens is provided, documents aren't truncated below that value.
    Instead, remove the last documents to fit the budget.
    """
    toks = tokenizer(docs, add_special_tokens=False).input_ids
    doc_lengths = [len(x) for x in toks]

    if min_doc_tokens:
        # remove documents such that the rest fit into the budget
        # this is to avoid very short documents.
        max_num_docs = max_inp_tokens // min_doc_tokens
        if len(docs) > max_num_docs:
            toks = toks[:max_num_docs]
            doc_lengths = doc_lengths[:max_num_docs]

    indexed_lst = sorted([(val, i) for i, val in enumerate(doc_lengths)])
    result = []
    remaining_sum = max_inp_tokens
    for i in range(len(indexed_lst)):
        avg = remaining_sum // (len(indexed_lst) - i)
        if indexed_lst[i][0] <= avg:
            result.append((indexed_lst[i][0], indexed_lst[i][1]))
            remaining_sum -= indexed_lst[i][0]
        else:
            result.append((avg, indexed_lst[i][1]))
            remaining_sum -= avg
    result.sort(key=lambda x: x[1])
    truncated_doc_lengths = [val for val, _ in result]
    # limits tokens in each document to the above values,
    # but only truncate at sentence boundaries
    truncated_docs = []
    for doc, doc_tok_limit in zip(docs, truncated_doc_lengths):
        sents = nltk.sent_tokenize(doc)
        truncated_doc = ""
        for sent in sents:
            if (
                len(
                    tokenizer.encode(
                        truncated_doc + " " + sent, add_special_tokens=False
                    )
                )
                > doc_tok_limit
            ):
                break
            truncated_doc += " " + sent
        truncated_docs += [truncated_doc]
    return truncated_docs


def preprocess_example(
    example: dict,
    tokenizer: AutoTokenizer,
    max_inp_tokens: int,
    doc_key: str,
    min_doc_tokens: int | None = None,
) -> dict:
    r"""
    Truncate documents to max_length, and return a list of documents.

    - If a list of list of documents is provided, budget equally to each list of docs.
        This is useful for timeline summarization datasets.
    - Within a list of topically related docs, truncate the longest documents first.
    """
    docs = example[doc_key]

    # check for minimum document length
    if isinstance(docs[0], list):
        if min_doc_tokens and (max_inp_tokens // len(docs)) < min_doc_tokens:
            # we have skip some timestamps to allow at least one document in each topic
            # keep most recent timestamps
            docs = docs[-1 * (max_inp_tokens // min_doc_tokens) :]
        truncated_docs = []
        for i in range(len(docs)):
            truncated_docs += truncate_helper(
                docs[i],
                max_inp_tokens=max_inp_tokens // len(docs),
                tokenizer=tokenizer,
                min_doc_tokens=min_doc_tokens,
            )
    elif isinstance(docs, list):
        truncated_docs = truncate_helper(
            docs,
            max_inp_tokens=max_inp_tokens,
            tokenizer=tokenizer,
            min_doc_tokens=min_doc_tokens,
        )
    # filter any empty docs
    truncated_docs = [doc for doc in truncated_docs if doc.strip() != ""]
    example[doc_key] = truncated_docs
    return example


def truncate_dataset(  # noqa: PLR0913
    dataset: Dataset,
    doc_key: str,
    max_length: int,
    tokenizer: AutoTokenizer,
    min_doc_tokens: int | None = None,
    debug: bool = False,
) -> Dataset:
    """Truncate HF dataset."""
    if debug:
        logger.warning("debug mode, using only 100 examples")
        dataset = dataset.select(range(100))

    return dataset.map(
        preprocess_example,
        batched=False,
        desc=f"truncating input to {max_length} tokens",
        load_from_cache_file=False,
        num_proc=1,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_inp_tokens": max_length,
            "doc_key": doc_key,
            "min_doc_tokens": min_doc_tokens,
        },
    )


def prepare_queries(dataset: Dataset, config: SummDataset) -> list[str]:
    """Prepare queries for a given dataset."""
    if getattr(config, "query_key", None) is None:
        return [config.default_query] * len(dataset)
    if getattr(config, "query_prompt", None) is None:
        return [ex[config.query_key] for ex in dataset]
    return [config.query_prompt.format(query=ex[config.query_key]) for ex in dataset]
