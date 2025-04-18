"""
Score system summaries.

Suports ROUGE, AutoACU (A3CU)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import pandas as pd
from datasets import load_from_disk
from loguru import logger

from config_utils import init_dataset_config, init_model_config
from metrics import compute_summ_metrics
from summarize import load_examples

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)


def score(  # noqa: PLR0913
    model_config_name: str,
    dataset_config_name: str,
    split: str,
    artifacts_dir: str,
    output_dir: str,
    tgt_pooling: str = "max",
    metrics: str = "rouge,a3cu",
) -> None:
    """Score system summaries."""
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)
    dataset_config = init_dataset_config(dataset_config_name, artifacts_dir)
    model_config = init_model_config(
        model_config_name, dataset_config_name, split, artifacts_dir, output_dir
    )

    # loading reference summaries
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
    tgt = dataset[dataset_config.summary_key]

    # loading system summaries
    pred_path = output_dir / f"{dataset_config_name}_{model_config_name}_{split}.jsonl"
    pred = []
    with pred_path.open() as rf:
        for line in rf:
            data = json.loads(line)
            pred += [data["pred"]]

    logger.info("pred: {}, tgt: {}", len(pred), len(tgt))

    scores, per_ex_scores = compute_summ_metrics(
        pred=pred,
        tgt=tgt,
        artifacts_dir=artifacts_dir,
        tgt_pooling=tgt_pooling,
        metrics=metrics,
    )

    # write corpus-level scores
    scores_path = (
        output_dir / f"{dataset_config_name}_{model_config_name}_{split}_scores.txt"
    )
    with scores_path.open("w") as wf:
        wf.write(
            pd.DataFrame.from_dict(scores, orient="index", columns=[model_config_name])
            .transpose()
            .to_string(),
        )
        wf.write("\n")
    # write per example scores
    per_ex_score_path = (
        output_dir
        / f"{dataset_config_name}_{model_config_name}_{split}_per_ex_scores.txt"
    )
    with per_ex_score_path.open("w") as wf:
        wf.write(
            pd.DataFrame.from_dict(per_ex_scores, orient="index")
            .transpose()
            .to_string(),
        )
        wf.write("\n")


if __name__ == "__main__":
    fire.Fire(score)
