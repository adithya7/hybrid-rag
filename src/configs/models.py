"""Model config at inference."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseModel:
    """Default model config."""

    output_dir: Path = None
    pred_path: Path = None

    # prompt
    prompt: str = (
        "{document}"
        "\n\n"
        "Question: {question}"
        "\n\n"
        "Answer the question based on the provided document. "
        "Be concise and directly address only the specific question asked. "
        "Limit your response to a maximum of {num_words} words."
        "\n\n"
    )

    # vLLM sampling params
    temperature: float = 0.5
    top_p: float = 1.0
    best_of: int = None
    seed: int = 43
    n_preds: int = 3  # we sample multiple summaries for each input


"""
Retrieval methods.
"""


@dataclass
class Flat_GTE_1p5B:
    """
    RAG with flat retrieval.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_GTE_1p5B"


@dataclass
class Flat_8K_GTE_1p5B:
    """
    RAG with flat retrieval, 8K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_8K_GTE_1p5B"


@dataclass
class Flat_16K_GTE_1p5B:
    """
    RAG with flat retrieval, 16K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_16K_GTE_1p5B"


@dataclass
class Flat_24K_GTE_1p5B:
    """
    RAG with flat retrieval, 24K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_24K_GTE_1p5B"


@dataclass
class Flat_32K_GTE_1p5B:
    """
    RAG with flat retrieval, 32K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_32K_GTE_1p5B"


@dataclass
class Flat_40K_GTE_1p5B:
    """
    RAG with flat retrieval, 40K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_40K_GTE_1p5B"


@dataclass
class Flat_48K_GTE_1p5B:
    """
    RAG with flat retrieval, 48K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_48K_GTE_1p5B"


@dataclass
class Flat_56K_GTE_1p5B:
    """
    RAG with flat retrieval, 56K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_56K_GTE_1p5B"


@dataclass
class Flat_64K_GTE_1p5B:
    """
    RAG with flat retrieval, 64K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_64K_GTE_1p5B"


@dataclass
class Flat_72K_GTE_1p5B:
    """
    RAG with flat retrieval, 72K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_72K_GTE_1p5B"


@dataclass
class Flat_80K_GTE_1p5B:
    """
    RAG with flat retrieval, 80K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """

    retriever: str = "Flat_80K_GTE_1p5B"


@dataclass
class Flat_8K_GTE_7B:
    """
    RAG with flat retrieval, 8K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_8K_GTE_7B"


@dataclass
class Flat_16K_GTE_7B:
    """
    RAG with flat retrieval, 16K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_16K_GTE_7B"


@dataclass
class Flat_24K_GTE_7B:
    """
    RAG with flat retrieval, 24K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_24K_GTE_7B"


@dataclass
class Flat_32K_GTE_7B:
    """
    RAG with flat retrieval, 32K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_32K_GTE_7B"


@dataclass
class Flat_40K_GTE_7B:
    """
    RAG with flat retrieval, 40K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_40K_GTE_7B"


@dataclass
class Flat_48K_GTE_7B:
    """
    RAG with flat retrieval, 48K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_48K_GTE_7B"


@dataclass
class Flat_56K_GTE_7B:
    """
    RAG with flat retrieval, 56K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_56K_GTE_7B"


@dataclass
class Flat_64K_GTE_7B:
    """
    RAG with flat retrieval, 64K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_64K_GTE_7B"


@dataclass
class Flat_72K_GTE_7B:
    """
    RAG with flat retrieval, 72K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_72K_GTE_7B"


@dataclass
class Flat_80K_GTE_7B:
    """
    RAG with flat retrieval, 80K max length.

    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """

    retriever: str = "Flat_80K_GTE_7B"


@dataclass
class Llama31(BaseModel):
    """Default inference config for Llama-3.1-based models."""

    max_length: int = 128 * 1024
    word2token_ratio: float = 1.145


@dataclass
class Llama32(Llama31):
    """Default inference config for Llama-3.2-based models."""


@dataclass
class Llama33(Llama31):
    """Default inference config for Llama-3.3-based models."""


@dataclass
class Llama31_8B(Llama31):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    Full context.
    """

    model_name_or_path: Path = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass
class Llama31_8B_Flat_8K_GTE_1p5B(Llama31_8B, Flat_8K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_8K_GTE_7B(Llama31_8B, Flat_8K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_16K_GTE_1p5B(Llama31_8B, Flat_16K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_16K_GTE_7B(Llama31_8B, Flat_16K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_24K_GTE_1p5B(Llama31_8B, Flat_24K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_24K_GTE_7B(Llama31_8B, Flat_24K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_32K_GTE_1p5B(Llama31_8B, Flat_32K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_32K_GTE_7B(Llama31_8B, Flat_32K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_40K_GTE_1p5B(Llama31_8B, Flat_40K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_40K_GTE_7B(Llama31_8B, Flat_40K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_48K_GTE_1p5B(Llama31_8B, Flat_48K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_48K_GTE_7B(Llama31_8B, Flat_48K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_56K_GTE_1p5B(Llama31_8B, Flat_56K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_56K_GTE_7B(Llama31_8B, Flat_56K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_64K_GTE_1p5B(Llama31_8B, Flat_64K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_64K_GTE_7B(Llama31_8B, Flat_64K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_72K_GTE_1p5B(Llama31_8B, Flat_72K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_72K_GTE_7B(Llama31_8B, Flat_72K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama31_8B_Flat_80K_GTE_1p5B(Llama31_8B, Flat_80K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama31_8B_Flat_80K_GTE_7B(Llama31_8B, Flat_80K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.1-8B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B(Llama32):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    full context.
    """

    model_name_or_path: Path = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass
class Llama32_1B_Flat_8K_GTE_1p5B(Llama32_1B, Flat_8K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_8K_GTE_7B(Llama32_1B, Flat_8K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_16K_GTE_1p5B(Llama32_1B, Flat_16K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_16K_GTE_7B(Llama32_1B, Flat_16K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_24K_GTE_1p5B(Llama32_1B, Flat_24K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_24K_GTE_7B(Llama32_1B, Flat_24K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_32K_GTE_1p5B(Llama32_1B, Flat_32K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_32K_GTE_7B(Llama32_1B, Flat_32K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_40K_GTE_1p5B(Llama32_1B, Flat_40K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_40K_GTE_7B(Llama32_1B, Flat_40K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_48K_GTE_1p5B(Llama32_1B, Flat_48K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_48K_GTE_7B(Llama32_1B, Flat_48K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_56K_GTE_1p5B(Llama32_1B, Flat_56K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_56K_GTE_7B(Llama32_1B, Flat_56K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_64K_GTE_1p5B(Llama32_1B, Flat_64K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_64K_GTE_7B(Llama32_1B, Flat_64K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_72K_GTE_1p5B(Llama32_1B, Flat_72K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_72K_GTE_7B(Llama32_1B, Flat_72K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_1B_Flat_80K_GTE_1p5B(Llama32_1B, Flat_80K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_1B_Flat_80K_GTE_7B(Llama32_1B, Flat_80K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-1B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B(Llama32):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    full context.
    """

    model_name_or_path: Path = "meta-llama/Llama-3.2-3B-Instruct"


@dataclass
class Llama32_3B_Flat_8K_GTE_1p5B(Llama32_3B, Flat_8K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_8K_GTE_7B(Llama32_3B, Flat_8K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_16K_GTE_1p5B(Llama32_3B, Flat_16K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_16K_GTE_7B(Llama32_3B, Flat_16K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_24K_GTE_1p5B(Llama32_3B, Flat_24K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_24K_GTE_7B(Llama32_3B, Flat_24K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_32K_GTE_1p5B(Llama32_3B, Flat_32K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_32K_GTE_7B(Llama32_3B, Flat_32K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_40K_GTE_1p5B(Llama32_3B, Flat_40K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_40K_GTE_7B(Llama32_3B, Flat_40K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_48K_GTE_1p5B(Llama32_3B, Flat_48K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_48K_GTE_7B(Llama32_3B, Flat_48K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_56K_GTE_1p5B(Llama32_3B, Flat_56K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_56K_GTE_7B(Llama32_3B, Flat_56K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_64K_GTE_1p5B(Llama32_3B, Flat_64K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_64K_GTE_7B(Llama32_3B, Flat_64K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_72K_GTE_1p5B(Llama32_3B, Flat_72K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_72K_GTE_7B(Llama32_3B, Flat_72K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama32_3B_Flat_80K_GTE_1p5B(Llama32_3B, Flat_80K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama32_3B_Flat_80K_GTE_7B(Llama32_3B, Flat_80K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.2-3B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B(Llama33):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    full context.
    fp8 quantization.
    """

    model_name_or_path: Path = "meta-llama/Llama-3.3-70B-Instruct"
    quantization: str = "fp8"


@dataclass
class Llama33_70B_Flat_8K_GTE_1p5B(Llama33_70B, Flat_8K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_8K_GTE_7B(Llama33_70B, Flat_8K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_16K_GTE_1p5B(Llama33_70B, Flat_16K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_16K_GTE_7B(Llama33_70B, Flat_16K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_24K_GTE_1p5B(Llama33_70B, Flat_24K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_24K_GTE_7B(Llama33_70B, Flat_24K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_32K_GTE_1p5B(Llama33_70B, Flat_32K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_32K_GTE_7B(Llama33_70B, Flat_32K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_40K_GTE_1p5B(Llama33_70B, Flat_40K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_40K_GTE_7B(Llama33_70B, Flat_40K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_48K_GTE_1p5B(Llama33_70B, Flat_48K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_48K_GTE_7B(Llama33_70B, Flat_48K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_56K_GTE_1p5B(Llama33_70B, Flat_56K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_56K_GTE_7B(Llama33_70B, Flat_56K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_64K_GTE_1p5B(Llama33_70B, Flat_64K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_64K_GTE_7B(Llama33_70B, Flat_64K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_72K_GTE_1p5B(Llama33_70B, Flat_72K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_72K_GTE_7B(Llama33_70B, Flat_72K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Llama33_70B_Flat_80K_GTE_1p5B(Llama33_70B, Flat_80K_GTE_1p5B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Llama33_70B_Flat_80K_GTE_7B(Llama33_70B, Flat_80K_GTE_7B):
    """
    summarizer: meta-llama/Llama-3.3-70B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


"""
Jamba 1.5 Mini
"""


@dataclass
class Jamba15Mini(BaseModel):
    """
    summarizer: ai21labs/AI21-Jamba-1.5-Mini.

    full context.
    """

    model_name_or_path: Path = "ai21labs/AI21-Jamba-1.5-Mini"
    max_length: int = 128 * 1024
    word2token_ratio: float = 1.219


@dataclass
class Qwen25(BaseModel):
    """Default inference config for Qwen2.5-based models."""

    word2token_ratio: float = 1.15
    max_length: int = 32 * 1024


@dataclass
class Qwen25_0p5B(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    full context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Qwen25_0p5B_Flat_8K_GTE_1p5B(Qwen25_0p5B, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_0p5B_Flat_8K_GTE_7B(Qwen25_0p5B, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_0p5B_Flat_16K_GTE_1p5B(Qwen25_0p5B, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_0p5B_Flat_16K_GTE_7B(Qwen25_0p5B, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_0p5B_Flat_24K_GTE_1p5B(Qwen25_0p5B, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_0p5B_Flat_24K_GTE_7B(Qwen25_0p5B, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_0p5B_Flat_32K_GTE_1p5B(Qwen25_0p5B, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_0p5B_Flat_32K_GTE_7B(Qwen25_0p5B, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-0.5B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_1p5B(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    full context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Qwen25_1p5B_Flat_8K_GTE_1p5B(Qwen25_1p5B, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_1p5B_Flat_8K_GTE_7B(Qwen25_1p5B, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_1p5B_Flat_16K_GTE_1p5B(Qwen25_1p5B, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_1p5B_Flat_16K_GTE_7B(Qwen25_1p5B, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_1p5B_Flat_24K_GTE_1p5B(Qwen25_1p5B, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_1p5B_Flat_24K_GTE_7B(Qwen25_1p5B, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_1p5B_Flat_32K_GTE_1p5B(Qwen25_1p5B, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_1p5B_Flat_32K_GTE_7B(Qwen25_1p5B, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-1.5B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_3B(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    full context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class Qwen25_3B_Flat_8K_GTE_1p5B(Qwen25_3B, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_3B_Flat_8K_GTE_7B(Qwen25_3B, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_3B_Flat_16K_GTE_1p5B(Qwen25_3B, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_3B_Flat_16K_GTE_7B(Qwen25_3B, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_3B_Flat_24K_GTE_1p5B(Qwen25_3B, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_3B_Flat_24K_GTE_7B(Qwen25_3B, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_3B_Flat_32K_GTE_1p5B(Qwen25_3B, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_3B_Flat_32K_GTE_7B(Qwen25_3B, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-3B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    default context length.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class Qwen25_7B_128K(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    full context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-7B-Instruct-128K"
    max_length: int = 128 * 1024


@dataclass
class Qwen25_7B_Flat_8K_GTE_1p5B(Qwen25_7B, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_8K_GTE_7B(Qwen25_7B, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_16K_GTE_1p5B(Qwen25_7B, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_16K_GTE_7B(Qwen25_7B, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_24K_GTE_1p5B(Qwen25_7B, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_24K_GTE_7B(Qwen25_7B, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_32K_GTE_1p5B(Qwen25_7B, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_32K_GTE_7B(Qwen25_7B, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_40K_GTE_1p5B(Qwen25_7B_128K, Flat_40K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_40K_GTE_7B(Qwen25_7B_128K, Flat_40K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_48K_GTE_1p5B(Qwen25_7B_128K, Flat_48K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_48K_GTE_7B(Qwen25_7B_128K, Flat_48K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_56K_GTE_1p5B(Qwen25_7B_128K, Flat_56K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_56K_GTE_7B(Qwen25_7B_128K, Flat_56K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_64K_GTE_1p5B(Qwen25_7B_128K, Flat_64K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


class Qwen25_7B_Flat_64K_GTE_7B(Qwen25_7B_128K, Flat_64K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_72K_GTE_1p5B(Qwen25_7B_128K, Flat_72K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_72K_GTE_7B(Qwen25_7B_128K, Flat_72K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_80K_GTE_1p5B(Qwen25_7B_128K, Flat_80K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_7B_Flat_80K_GTE_7B(Qwen25_7B_128K, Flat_80K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Qwen25_7B_1M(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    full (but limited) context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-7B-Instruct-1M"
    max_length: int = 128 * 1024  # we limit to 128k for fair comparison


@dataclass
class Qwen25_7B_1M_Flat_8K_GTE_1p5B(Qwen25_7B_1M, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_8K_GTE_7B(Qwen25_7B_1M, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_16K_GTE_1p5B(Qwen25_7B_1M, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_16K_GTE_7B(Qwen25_7B_1M, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_24K_GTE_1p5B(Qwen25_7B_1M, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_24K_GTE_7B(Qwen25_7B_1M, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_32K_GTE_1p5B(Qwen25_7B_1M, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_32K_GTE_7B(Qwen25_7B_1M, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_40K_GTE_1p5B(Qwen25_7B_1M, Flat_40K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_40K_GTE_7B(Qwen25_7B_1M, Flat_40K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_48K_GTE_1p5B(Qwen25_7B_1M, Flat_48K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_48K_GTE_7B(Qwen25_7B_1M, Flat_48K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_56K_GTE_1p5B(Qwen25_7B_1M, Flat_56K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_56K_GTE_7B(Qwen25_7B_1M, Flat_56K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_64K_GTE_1p5B(Qwen25_7B_1M, Flat_64K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_64K_GTE_7B(Qwen25_7B_1M, Flat_64K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_72K_GTE_1p5B(Qwen25_7B_1M, Flat_72K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_72K_GTE_7B(Qwen25_7B_1M, Flat_72K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_80K_GTE_1p5B(Qwen25_7B_1M, Flat_80K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_7B_1M_Flat_80K_GTE_7B(Qwen25_7B_1M, Flat_80K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-7B-Instruct-1M.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    default context length.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-14B-Instruct"


@dataclass
class Qwen25_14B_128K(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    full context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-14B-Instruct-128K"
    max_length: int = 128 * 1024


@dataclass
class Qwen25_14B_Flat_8K_GTE_1p5B(Qwen25_14B, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_8K_GTE_7B(Qwen25_14B, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_16K_GTE_1p5B(Qwen25_14B, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_16K_GTE_7B(Qwen25_14B, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_24K_GTE_1p5B(Qwen25_14B, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_24K_GTE_7B(Qwen25_14B, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_32K_GTE_1p5B(Qwen25_14B, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_32K_GTE_7B(Qwen25_14B, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_40K_GTE_1p5B(Qwen25_14B_128K, Flat_40K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_40K_GTE_7B(Qwen25_14B_128K, Flat_40K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_48K_GTE_1p5B(Qwen25_14B_128K, Flat_48K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_48K_GTE_7B(Qwen25_14B_128K, Flat_48K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_56K_GTE_1p5B(Qwen25_14B_128K, Flat_56K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_56K_GTE_7B(Qwen25_14B_128K, Flat_56K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_Flat_64K_GTE_1p5B(Qwen25_14B_128K, Flat_64K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_Flat_64K_GTE_7B(Qwen25_14B_128K, Flat_64K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_Flat_72K_GTE_1p5B(Qwen25_14B_128K, Flat_72K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_Flat_72K_GTE_7B(Qwen25_14B_128K, Flat_72K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_Flat_80K_GTE_1p5B(Qwen25_14B_128K, Flat_80K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_Flat_80K_GTE_7B(Qwen25_14B_128K, Flat_80K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_1M(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    full (but limited) context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-14B-Instruct-1M"
    max_length: int = 128 * 1024  # we limit to 128k for fair comparison


@dataclass
class Qwen25_14B_1M_Flat_8K_GTE_1p5B(Qwen25_14B_1M, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_8K_GTE_7B(Qwen25_14B_1M, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_16K_GTE_1p5B(Qwen25_14B_1M, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_16K_GTE_7B(Qwen25_14B_1M, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_24K_GTE_1p5B(Qwen25_14B_1M, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_24K_GTE_7B(Qwen25_14B_1M, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_32K_GTE_1p5B(Qwen25_14B_1M, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_32K_GTE_7B(Qwen25_14B_1M, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_40K_GTE_1p5B(Qwen25_14B_1M, Flat_40K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_40K_GTE_7B(Qwen25_14B_1M, Flat_40K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_48K_GTE_1p5B(Qwen25_14B_1M, Flat_48K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_48K_GTE_7B(Qwen25_14B_1M, Flat_48K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_56K_GTE_1p5B(Qwen25_14B_1M, Flat_56K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_56K_GTE_7B(Qwen25_14B_1M, Flat_56K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_64K_GTE_1p5B(Qwen25_14B_1M, Flat_64K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_64K_GTE_7B(Qwen25_14B_1M, Flat_64K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_72K_GTE_1p5B(Qwen25_14B_1M, Flat_72K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_72K_GTE_7B(Qwen25_14B_1M, Flat_72K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_80K_GTE_1p5B(Qwen25_14B_1M, Flat_80K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_14B_1M_Flat_80K_GTE_7B(Qwen25_14B_1M, Flat_80K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-14B-Instruct-1M.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_32B(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    default context length.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-32B-Instruct"


@dataclass
class Qwen25_32B_128K(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    full context.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-32B-Instruct-128K"
    max_length: int = 128 * 1024


@dataclass
class Qwen25_32B_Flat_GTE_1p5B(Qwen25_32B, Flat_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Qwen25_32B_Flat_8K_GTE_1p5B(Qwen25_32B, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_8K_GTE_7B(Qwen25_32B, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_16K_GTE_1p5B(Qwen25_32B, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_16K_GTE_7B(Qwen25_32B, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_24K_GTE_1p5B(Qwen25_32B, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_24K_GTE_7B(Qwen25_32B, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_32K_GTE_1p5B(Qwen25_32B, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_32K_GTE_7B(Qwen25_32B, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_40K_GTE_1p5B(Qwen25_32B_128K, Flat_40K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_40K_GTE_7B(Qwen25_32B_128K, Flat_40K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_48K_GTE_1p5B(Qwen25_32B_128K, Flat_48K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_48K_GTE_7B(Qwen25_32B_128K, Flat_48K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_56K_GTE_1p5B(Qwen25_32B_128K, Flat_56K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_56K_GTE_7B(Qwen25_32B_128K, Flat_56K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_32B_Flat_64K_GTE_1p5B(Qwen25_32B_128K, Flat_64K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_32B_Flat_64K_GTE_7B(Qwen25_32B_128K, Flat_64K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_32B_Flat_72K_GTE_1p5B(Qwen25_32B_128K, Flat_72K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_32B_Flat_72K_GTE_7B(Qwen25_32B_128K, Flat_72K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_32B_Flat_80K_GTE_1p5B(Qwen25_32B_128K, Flat_80K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_32B_Flat_80K_GTE_7B(Qwen25_32B_128K, Flat_80K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-32B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_72B(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    default context length.
    fp8 quantization.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-72B-Instruct"
    quantization: str = "fp8"


@dataclass
class Qwen25_72B_128K(Qwen25):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    full context.
    fp8 quantization.
    """

    model_name_or_path: Path = "Qwen/Qwen2.5-72B-Instruct-128K"
    quantization: str = "fp8"
    max_length: int = 128 * 1024


@dataclass
class Qwen25_72B_Flat_8K_GTE_1p5B(Qwen25_72B, Flat_8K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_8K_GTE_7B(Qwen25_72B, Flat_8K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_16K_GTE_1p5B(Qwen25_72B, Flat_16K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_16K_GTE_7B(Qwen25_72B, Flat_16K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_24K_GTE_1p5B(Qwen25_72B, Flat_24K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_24K_GTE_7B(Qwen25_72B, Flat_24K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_32K_GTE_1p5B(Qwen25_72B, Flat_32K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_32K_GTE_7B(Qwen25_72B, Flat_32K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_40K_GTE_1p5B(Qwen25_72B_128K, Flat_40K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_40K_GTE_7B(Qwen25_72B_128K, Flat_40K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_48K_GTE_1p5B(Qwen25_72B_128K, Flat_48K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_48K_GTE_7B(Qwen25_72B_128K, Flat_48K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_56K_GTE_1p5B(Qwen25_72B_128K, Flat_56K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_56K_GTE_7B(Qwen25_72B_128K, Flat_56K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 without YARN.
    """


@dataclass
class Qwen25_72B_Flat_64K_GTE_1p5B(Qwen25_72B_128K, Flat_64K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_72B_Flat_64K_GTE_7B(Qwen25_72B_128K, Flat_64K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_72B_Flat_72K_GTE_1p5B(Qwen25_72B_128K, Flat_72K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_72B_Flat_72K_GTE_7B(Qwen25_72B_128K, Flat_72K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_72B_Flat_80K_GTE_1p5B(Qwen25_72B_128K, Flat_80K_GTE_1p5B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Qwen25_72B_Flat_80K_GTE_7B(Qwen25_72B_128K, Flat_80K_GTE_7B):
    """
    summarizer: Qwen/Qwen2.5-72B-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.

    use Qwen25 with YARN.
    """


@dataclass
class Phi3(BaseModel):
    """Default inference config for Phi-3-based models."""

    word2token_ratio: float = 1.3
    max_length: int = 128 * 1024


@dataclass
class Phi3Mini(Phi3):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    full context.
    """

    model_name_or_path: Path = "microsoft/Phi-3-mini-128k-instruct"


@dataclass
class Phi3Mini_Flat_8K_GTE_1p5B(Phi3Mini, Flat_8K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_8K_GTE_7B(Phi3Mini, Flat_8K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_16K_GTE_1p5B(Phi3Mini, Flat_16K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_16K_GTE_7B(Phi3Mini, Flat_16K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_24K_GTE_1p5B(Phi3Mini, Flat_24K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_24K_GTE_7B(Phi3Mini, Flat_24K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_32K_GTE_1p5B(Phi3Mini, Flat_32K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_32K_GTE_7B(Phi3Mini, Flat_32K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_40K_GTE_1p5B(Phi3Mini, Flat_40K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_40K_GTE_7B(Phi3Mini, Flat_40K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_48K_GTE_1p5B(Phi3Mini, Flat_48K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_48K_GTE_7B(Phi3Mini, Flat_48K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_56K_GTE_1p5B(Phi3Mini, Flat_56K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_56K_GTE_7B(Phi3Mini, Flat_56K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_64K_GTE_1p5B(Phi3Mini, Flat_64K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_64K_GTE_7B(Phi3Mini, Flat_64K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_72K_GTE_1p5B(Phi3Mini, Flat_72K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_72K_GTE_7B(Phi3Mini, Flat_72K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Mini_Flat_80K_GTE_1p5B(Phi3Mini, Flat_80K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Mini_Flat_80K_GTE_7B(Phi3Mini, Flat_80K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-mini-128k-instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small(Phi3):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    full context.
    """

    model_name_or_path: Path = "microsoft/Phi-3-small-128k-instruct"
    trust_remote_code: bool = True
    enable_chunked_prefill: bool = False


@dataclass
class Phi3Small_Flat_8K_GTE_1p5B(Phi3Small, Flat_8K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_8K_GTE_7B(Phi3Small, Flat_8K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_16K_GTE_1p5B(Phi3Small, Flat_16K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_16K_GTE_7B(Phi3Small, Flat_16K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_24K_GTE_1p5B(Phi3Small, Flat_24K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_24K_GTE_7B(Phi3Small, Flat_24K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_32K_GTE_1p5B(Phi3Small, Flat_32K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_32K_GTE_7B(Phi3Small, Flat_32K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_40K_GTE_1p5B(Phi3Small, Flat_40K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_40K_GTE_7B(Phi3Small, Flat_40K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_48K_GTE_1p5B(Phi3Small, Flat_48K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_48K_GTE_7B(Phi3Small, Flat_48K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_56K_GTE_1p5B(Phi3Small, Flat_56K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_56K_GTE_7B(Phi3Small, Flat_56K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_64K_GTE_1p5B(Phi3Small, Flat_64K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_64K_GTE_7B(Phi3Small, Flat_64K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_72K_GTE_1p5B(Phi3Small, Flat_72K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_72K_GTE_7B(Phi3Small, Flat_72K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Small_Flat_80K_GTE_1p5B(Phi3Small, Flat_80K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Small_Flat_80K_GTE_7B(Phi3Small, Flat_80K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-small-128k-instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium(Phi3):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    full context.
    """

    model_name_or_path: Path = "microsoft/Phi-3-medium-128k-instruct"


@dataclass
class Phi3Medium_Flat_8K_GTE_1p5B(Phi3Medium, Flat_8K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_8K_GTE_7B(Phi3Medium, Flat_8K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_16K_GTE_1p5B(Phi3Medium, Flat_16K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_16K_GTE_7B(Phi3Medium, Flat_16K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_24K_GTE_1p5B(Phi3Medium, Flat_24K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_24K_GTE_7B(Phi3Medium, Flat_24K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_32K_GTE_1p5B(Phi3Medium, Flat_32K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_32K_GTE_7B(Phi3Medium, Flat_32K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_40K_GTE_1p5B(Phi3Medium, Flat_40K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_40K_GTE_7B(Phi3Medium, Flat_40K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_48K_GTE_1p5B(Phi3Medium, Flat_48K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_48K_GTE_7B(Phi3Medium, Flat_48K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_56K_GTE_1p5B(Phi3Medium, Flat_56K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_56K_GTE_7B(Phi3Medium, Flat_56K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_64K_GTE_1p5B(Phi3Medium, Flat_64K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_64K_GTE_7B(Phi3Medium, Flat_64K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_72K_GTE_1p5B(Phi3Medium, Flat_72K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_72K_GTE_7B(Phi3Medium, Flat_72K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class Phi3Medium_Flat_80K_GTE_1p5B(Phi3Medium, Flat_80K_GTE_1p5B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class Phi3Medium_Flat_80K_GTE_7B(Phi3Medium, Flat_80K_GTE_7B):
    """
    summarizer: microsoft/Phi-3-medium-128k-instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong(BaseModel):
    """Default inference config for ProLong-based models."""

    word2token_ratio: float = 1.145


@dataclass
class ProLong64K(ProLong):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    full context.
    """

    model_name_or_path: Path = "princeton-nlp/Llama-3-8B-ProLong-64k-Instruct"
    max_length: int = 64 * 1024


@dataclass
class ProLong64K_Flat_8K_GTE_1p5B(ProLong64K, Flat_8K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_8K_GTE_7B(ProLong64K, Flat_8K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong64K_Flat_16K_GTE_1p5B(ProLong64K, Flat_16K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_16K_GTE_7B(ProLong64K, Flat_16K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong64K_Flat_24K_GTE_1p5B(ProLong64K, Flat_24K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_24K_GTE_7B(ProLong64K, Flat_24K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong64K_Flat_32K_GTE_1p5B(ProLong64K, Flat_32K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_32K_GTE_7B(ProLong64K, Flat_32K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong64K_Flat_40K_GTE_1p5B(ProLong64K, Flat_40K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_40K_GTE_7B(ProLong64K, Flat_40K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong64K_Flat_48K_GTE_1p5B(ProLong64K, Flat_48K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_48K_GTE_7B(ProLong64K, Flat_48K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong64K_Flat_56K_GTE_1p5B(ProLong64K, Flat_56K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_56K_GTE_7B(ProLong64K, Flat_56K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong64K_Flat_64K_GTE_1p5B(ProLong64K, Flat_64K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong64K_Flat_64K_GTE_7B(ProLong64K, Flat_64K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K(ProLong):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-512k-Instruct.

    full (but limited) context.
    """

    model_name_or_path: Path = "princeton-nlp/Llama-3-8B-ProLong-512k-Instruct"
    max_length: int = 128 * 1024  # we limit to 128k for fair comparison


@dataclass
class ProLong512K_Flat_8K_GTE_1p5B(ProLong512K, Flat_8K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_8K_GTE_7B(ProLong512K, Flat_8K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 8K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_16K_GTE_1p5B(ProLong512K, Flat_16K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_16K_GTE_7B(ProLong512K, Flat_16K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 16K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_24K_GTE_1p5B(ProLong512K, Flat_24K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_24K_GTE_7B(ProLong512K, Flat_24K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 24K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_32K_GTE_1p5B(ProLong512K, Flat_32K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_32K_GTE_7B(ProLong512K, Flat_32K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 32K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_40K_GTE_1p5B(ProLong512K, Flat_40K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_40K_GTE_7B(ProLong512K, Flat_40K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 40K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_48K_GTE_1p5B(ProLong512K, Flat_48K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_48K_GTE_7B(ProLong512K, Flat_48K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 48K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_56K_GTE_1p5B(ProLong512K, Flat_56K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_56K_GTE_7B(ProLong512K, Flat_56K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 56K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_64K_GTE_1p5B(ProLong512K, Flat_64K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_64K_GTE_7B(ProLong512K, Flat_64K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 64K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_72K_GTE_1p5B(ProLong512K, Flat_72K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_72K_GTE_7B(ProLong512K, Flat_72K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 72K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """


@dataclass
class ProLong512K_Flat_80K_GTE_1p5B(ProLong512K, Flat_80K_GTE_1p5B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-1.5B-instruct.
    """


@dataclass
class ProLong512K_Flat_80K_GTE_7B(ProLong512K, Flat_80K_GTE_7B):
    """
    summarizer: princeton-nlp/Llama-3-8B-ProLong-64k-Instruct.

    RAG with flat retrieval, 80K max length.
    embeddings: Alibaba-NLP/gte-Qwen2-7B-instruct.
    """
