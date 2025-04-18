"""Retriever systems."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseRetrieverConfig:
    """Base class for retrievers."""

    retriever_class: str
    output_dir: Path = None
    max_segment_tokens: int = 1024  # max length of each retrieval unit


@dataclass
class Flat(BaseRetrieverConfig):
    """Flat retriever."""

    retriever_class: str = "Flat"


@dataclass
class Flat_8K(Flat):
    """Flat retriever with 8K max length."""

    max_retrieved_tokens: int = 8 * 1024


@dataclass
class Flat_16K(Flat):
    """Flat retriever with 16K max length."""

    max_retrieved_tokens: int = 16 * 1024


@dataclass
class Flat_24K(Flat):
    """Flat retriever with 24K max length."""

    max_retrieved_tokens: int = 24 * 1024


@dataclass
class Flat_32K(Flat):
    """Flat retriever with 32K max length."""

    max_retrieved_tokens: int = 32 * 1024


@dataclass
class Flat_40K(Flat):
    """Flat retriever with 40K max length."""

    max_retrieved_tokens: int = 40 * 1024


@dataclass
class Flat_48K(Flat):
    """Flat retriever with 48K max length."""

    max_retrieved_tokens: int = 48 * 1024


@dataclass
class Flat_56K(Flat):
    """Flat retriever with 56K max length."""

    max_retrieved_tokens: int = 56 * 1024


@dataclass
class Flat_64K(Flat):
    """Flat retriever with 64K max length."""

    max_retrieved_tokens: int = 64 * 1024


@dataclass
class Flat_72K(Flat):
    """Flat retriever with 72K max length."""

    max_retrieved_tokens: int = 72 * 1024


@dataclass
class Flat_80K(Flat):
    """Flat retriever with 80K max length."""

    max_retrieved_tokens: int = 80 * 1024


"""
Embedding models.
"""


@dataclass
class GTE_1p5B:
    """
    GTE-1.5B embeddings.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    embedder: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    trust_remote_code: bool = True
    query_prompt: str = (
        "Instruct: "
        "Given a web search query, "
        "retrieve relevant passages that answer the query."
        "\n"
        "Query: {query}"
    )


@dataclass
class GTE_7B:
    """
    GTE-7B embeddings.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    embedder: str = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    trust_remote_code: bool = True
    query_prompt: str = (
        "Instruct: "
        "Given a web search query, "
        "retrieve relevant passages that answer the query."
        "\n"
        "Query: {query}"
    )


"""
Custom embedders for the retrievers.

We experiment with GTE embeddings (1.5B, 7B)
"""


@dataclass
class Flat_GTE_1p5B(Flat, GTE_1p5B):
    """
    Flat retriever.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_8K_GTE_1p5B(Flat_8K, GTE_1p5B):
    """
    Flat retriever with 8K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_16K_GTE_1p5B(Flat_16K, GTE_1p5B):
    """
    Flat retriever with 16K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_24K_GTE_1p5B(Flat_24K, GTE_1p5B):
    """
    Flat retriever with 24K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_32K_GTE_1p5B(Flat_32K, GTE_1p5B):
    """
    Flat retriever with 32K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_40K_GTE_1p5B(Flat_40K, GTE_1p5B):
    """
    Flat retriever with 40K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_48K_GTE_1p5B(Flat_48K, GTE_1p5B):
    """
    Flat retriever with 48K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_56K_GTE_1p5B(Flat_56K, GTE_1p5B):
    """
    Flat retriever with 56K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_64K_GTE_1p5B(Flat_64K, GTE_1p5B):
    """
    Flat retriever with 64K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_72K_GTE_1p5B(Flat_72K, GTE_1p5B):
    """
    Flat retriever with 72K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_80K_GTE_1p5B(Flat_80K, GTE_1p5B):
    """
    Flat retriever with 80K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Flat_GTE_7B(Flat, GTE_7B):
    """
    Flat retriever.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_8K_GTE_7B(Flat_8K, GTE_7B):
    """
    Flat retriever with 8K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_16K_GTE_7B(Flat_16K, GTE_7B):
    """
    Flat retriever with 16K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_24K_GTE_7B(Flat_24K, GTE_7B):
    """
    Flat retriever with 24K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_32K_GTE_7B(Flat_32K, GTE_7B):
    """
    Flat retriever with 32K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_40K_GTE_7B(Flat_40K, GTE_7B):
    """
    Flat retriever with 40K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_48K_GTE_7B(Flat_48K, GTE_7B):
    """
    Flat retriever with 48K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_56K_GTE_7B(Flat_56K, GTE_7B):
    """
    Flat retriever with 56K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_64K_GTE_7B(Flat_64K, GTE_7B):
    """
    Flat retriever with 64K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_72K_GTE_7B(Flat_72K, GTE_7B):
    """
    Flat retriever with 72K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Flat_80K_GTE_7B(Flat_80K, GTE_7B):
    """
    Flat retriever with 80K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """
