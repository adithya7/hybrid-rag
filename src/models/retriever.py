"""Retrieval models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import nltk
import numpy as np
import tiktoken
from vllm import LLM

if TYPE_CHECKING:
    from configs.retriever import BaseRetrieverConfig

tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count number of tokens, using OpenAI's tokenizer."""
    return len(tiktoken_tokenizer.encode(text))


class Retriever(ABC):
    """Retriever class."""

    def __init__(self, config: BaseRetrieverConfig) -> None:
        """Initialize retriever."""
        self.config = config
        self.init_model()

    @abstractmethod
    def init_model(self) -> None:
        """Initialize model(s)."""

    @abstractmethod
    def retrieve_docs(
        self, docs: list[list[str]], queries: list[str]
    ) -> list[list[str]]:
        """Retrieve relevant documents."""

    def select_docs(
        self,
        docs: list[str],
        scores: list[float],
        max_tokens: int,
    ) -> list[str]:
        """
        Select documents based on scores.

        Retain order of input documents.
        """
        # pick segments with highest scores
        # we use OpenAI's tokenizer to count tokens
        # this ensures same number of words for any summarizer
        curr_token_count = 0
        selected_docs = [False] * len(docs)
        for idx in np.argsort(scores)[::-1]:
            doc_token_count = count_tokens(docs[idx])
            if (curr_token_count + doc_token_count) > max_tokens:
                break
            selected_docs[idx] = True
            curr_token_count += doc_token_count
        return [docs[idx] for idx in range(len(docs)) if selected_docs[idx]]

    def get_bm25_scores(
        self,
        docs: list[str],
        query: str,
    ) -> list[float]:
        """Get BM25 scores for documents."""
        from rank_bm25 import BM25Okapi

        docs_toks = [nltk.word_tokenize(doc) for doc in docs]
        bm25 = BM25Okapi(docs_toks)
        return bm25.get_scores(nltk.word_tokenize(query))


class BM25Retriever(Retriever):
    """BM25 based retrieval."""

    def __init__(self, config: BaseRetrieverConfig) -> None:
        """Initialize retriever."""
        super().__init__(config)

    def init_model(self) -> None:
        """Initialize model."""

    def retrieve_docs(
        self,
        docs: list[list[str]],
        queries: list[str],
        max_retrieved_tokens: int,
    ) -> list[list[str]]:
        """Retrieve relevant documents using BM25."""
        selected_docs = []
        for idx in range(len(docs)):
            scores = self.get_bm25_scores(docs[idx], queries[idx])
            selected_docs += [self.select_docs(docs[idx], scores, max_retrieved_tokens)]
        return selected_docs


class FlatRetriever(Retriever):
    """Retriever models from huggingface."""

    def __init__(self, config: BaseRetrieverConfig) -> None:
        """Initialize retriever."""
        super().__init__(config)

    def init_model(self) -> None:
        """Initialize embedder."""
        # use vLLM
        self.embedder = LLM(
            model=self.config.embedder,
            task="embed",
            trust_remote_code=self.config.trust_remote_code,
        )

    def retrieve_docs(
        self,
        docs: list[list[str]],
        queries: list[str],
        max_retrieved_tokens: int,
    ) -> list[list[str]]:
        """
        Retrieve relevant documents. Select documents based on relevance to query.

        Batched retrieval.
        """
        assert len(docs) == len(queries)
        # if query prompt and segment prompts are provided, use them.
        query_prompts = queries
        if hasattr(self.config, "query_prompt"):
            query_prompts = [
                self.config.query_prompt.format(query=query) for query in queries
            ]
        doc_prompts = docs
        if hasattr(self.config, "segment_prompt"):
            for idx in range(len(docs)):
                doc_prompts[idx] = [
                    self.config.segment_prompt.format(doc=doc) for doc in docs[idx]
                ]

        # generate query embeddings
        query_embeddings = self.embedder.embed(query_prompts, use_tqdm=False)
        query_embeddings = np.array(
            [output.outputs.embedding for output in query_embeddings]
        )
        # generate document embeddings
        # first flatten the list of documents
        # compute embeddings
        # reshape to the original list of list of documents
        doc_embeddings = self.embedder.embed(
            [subitem for item in doc_prompts for subitem in item], use_tqdm=False
        )
        doc_embeddings = np.array(
            [output.outputs.embedding for output in doc_embeddings]
        )
        # use the length of each item in doc_prompts to reshape doc_embeddings
        doc_embeddings = np.split(
            doc_embeddings, np.cumsum([len(item) for item in doc_prompts])[:-1]
        )
        # compute scores
        selected_docs = []
        for idx in range(len(docs)):
            sim_scores = [query_embeddings[idx] @ d.T for d in doc_embeddings[idx]]
            if getattr(self.config, "bm25_fusion", False):
                # reciprocal rank fusion of bm25 and embedding similarity scores
                bm25_scores = self.get_bm25_scores(docs[idx], queries[idx])
                # get ranks from embedding scores and bm25 scores
                embedding_ranks = np.argsort(np.argsort(sim_scores)[::-1])
                bm25_ranks = np.argsort(np.argsort(bm25_scores)[::-1])
                # use reciprocal rank fusion
                scores = [
                    (1 / (60 + r1 + 1)) + (1 / (60 + r2 + 1))
                    for r1, r2 in zip(embedding_ranks, bm25_ranks)
                ]
            else:
                scores = sim_scores
            # select documents up to max_retrieved_tokens
            selected_docs += [
                self.select_docs(
                    docs[idx],
                    scores,
                    max_retrieved_tokens,
                )
            ]
        return selected_docs
