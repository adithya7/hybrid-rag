"""Config file for datasets."""

from dataclasses import dataclass


@dataclass
class SummDataset:
    """Default config for summarization datasets."""

    # default keys
    doc_key: str = "document"
    summary_key: str = "summary"
    query_key: str = "query"

    # default query for datasets that don't have one
    default_query: str = "Generate a summary of the document."

    min_doc_length: int = 128


@dataclass
class SummHay(SummDataset):
    """
    Summary of a haystack dataset from Laban et al., 2024.

    https://arxiv.org/abs/2407.01370
    """

    path: str = "misc/summhay/summhay.py"
    name: str = "summhay"
    max_summary_words: int = 245  # 80th percentile nltk tokens (test set)


@dataclass
class SummHayOracle(SummHay):
    """SummHay datasets with oracle documents."""

    name: str = "summhay_oracle"


@dataclass
class SummHay_SysPool_Silver(SummHay):
    """
    SummHay dataset.

    Using a pool of system summaries instead of gold reference summary.
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True


@dataclass
class SummHay_5SysPool_Silver(SummHay):
    """
    SummHay dataset.

    Using a pool of system summaries instead of gold reference summary.
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True


@dataclass
class SummHay_5SysPool10p_Silver(SummHay):
    """
    SummHay dataset.

    Using a pool of system summaries instead of gold reference summary. (10%)
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True


@dataclass
class SummHay_5SysPool50p_Silver(SummHay):
    """
    SummHay dataset.

    Using a pool of system summaries instead of gold reference summary. (50%)
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True


@dataclass
class SummHay_5SysPool75p_Silver(SummHay):
    """
    SummHay dataset.

    Using a pool of system summaries instead of gold reference summary. (75%)
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True


@dataclass
class SummHay_5SysPool100p_Silver(SummHay):
    """
    SummHay dataset.

    Using a pool of system summaries instead of gold reference summary. (100%)
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True


@dataclass
class SummHay_Llama33_70B_Silver(SummHay):
    """
    SummHay dataset.

    Using Llama33_70B system summaries instead of gold reference summary.
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True


@dataclass
class SummHay_Qwen25_72B_128K_Silver(SummHay):
    """
    SummHay dataset.

    Using Qwen25_72B_128K system summaries instead of gold reference summary.
    """

    path: str = None
    name: str = None
    load_from_disk: bool = True
