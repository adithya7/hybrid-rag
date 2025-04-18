"""vLLM utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from transformers import AutoTokenizer

    from configs.datasets import SummDataset
    from configs.models import BaseModel


def init_vllm(config: BaseModel, num_gpus: int) -> LLM:
    """Initialize vLLM model."""
    return LLM(
        model=config.model_name_or_path,
        dtype="bfloat16",
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        max_model_len=config.max_length,
        quantization=getattr(config, "quantization", None),
        load_format=getattr(config, "load_format", "auto"),
        enable_chunked_prefill=getattr(config, "enable_chunked_prefill", True),
    )


def check_input_type(docs: list[str]) -> bool:
    """Check if the input is a list of strings."""
    return isinstance(docs, list) and all(isinstance(item, str) for item in docs)


def get_prompt_token_ids(
    docs: list[str],
    queries: list[str],
    tokenizer: AutoTokenizer,
    model_config: BaseModel,
    dataset_config: SummDataset,
) -> list[list[int]]:
    """Prepare input for each example, and tokenize."""
    if not check_input_type(docs):
        msg = "docs should be a list of strings"
        raise TypeError(msg)
    # prepare input for each example
    # concatenate documents within each example
    # use default chat template
    # include documents in the message
    prompts = [
        model_config.prompt.format(
            document=ex_docs,
            num_words=dataset_config.max_summary_words,
            question=query,
        )
        for ex_docs, query in zip(docs, queries)
    ]
    # add model specific chat template
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def predict_vllm(  # noqa: PLR0913
    model: LLM,
    docs: list[str],
    queries: list[str],
    tokenizer: AutoTokenizer,
    model_config: BaseModel,
    dataset_config: SummDataset,
) -> list[str]:
    """Predict using vLLM."""
    sampling_params = SamplingParams(
        temperature=model_config.temperature,
        top_p=model_config.top_p,
        best_of=model_config.best_of,
        seed=model_config.seed,
        n=model_config.n_preds,
        max_tokens=dataset_config.max_summary_tokens,
    )
    prompt_token_ids = get_prompt_token_ids(
        docs=docs,
        queries=queries,
        tokenizer=tokenizer,
        model_config=model_config,
        dataset_config=dataset_config,
    )
    outputs = model.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    return [[pred.text for pred in output.outputs] for output in outputs]
