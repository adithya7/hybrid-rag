"""Gemini-related utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm

from client_api import APIClient

if TYPE_CHECKING:
    from configs.datasets import SummDataset
    from configs.models import BaseModel


def init_api(config: BaseModel) -> APIClient:
    """Initialize API client."""
    return APIClient(
        api=config.api,
        key_path=config.key_path,
        model=config.model_name_or_path,
    )


def predict_api(  # noqa: PLR0913
    client: APIClient,
    docs: list[list[str]],
    queries: list[str],
    model_config: BaseModel,
    dataset_config: SummDataset,
    temperature: float = 0.5,
) -> str:
    """Prediction using API-based models."""
    prompts = [
        model_config.prompt.format(
            document="\n".join(ex_docs),
            num_words=dataset_config.max_summary_words,
            question=query,
        )
        for ex_docs, query in zip(docs, queries)
    ]
    outputs = []
    for prompt in tqdm(prompts, desc="generating summaries", colour="green"):
        outputs += [
            client.obtain_response(
                prompt=prompt,
                max_tokens=dataset_config.max_summary_tokens,
                temperature=temperature,
            )
        ]
    return outputs
