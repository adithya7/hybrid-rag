"""Metrics for summary evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import evaluate
import nltk
import numpy as np
from autoacu import A3CU
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path


def postprocess_text(
    preds: list[str],
    labels: list[str] | list[list[str]],
) -> tuple[list[str], list[str] | list[list[str]]]:
    """Sentence segment text for rougeLSum computation."""
    # rougeLSum expects newline after each sentence
    if isinstance(preds[0], list):
        preds = [
            ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in _item]
            for _item in preds
        ]
    else:
        preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]

    if isinstance(labels[0], list):
        labels = [
            ["\n".join(nltk.sent_tokenize(label.strip())) for label in _item]
            for _item in labels
        ]
    else:
        labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]

    return preds, labels


def get_scores(
    pred: list[str],
    tgt: list[str] | list[list[str]],
    metrics: str,
    a3cu: A3CU | None,
    tgt_pooling: str,
) -> tuple[dict, dict]:
    """Compute scores for one set of predictions."""
    scores = {}
    per_ex_scores = defaultdict(list)

    # ROUGE
    if "rouge" in metrics.split(","):
        logger.info("computing rouge...")
        rouge_metric = evaluate.load("rouge")
        # aggregate scores
        rouge_scores = rouge_metric.compute(
            predictions=pred,
            references=tgt,
            use_stemmer=True,
        )
        # geometric_mean is inspired by Unlimiformer's use of SCROLLS metric
        # https://huggingface.co/datasets/tau/scrolls/blob/main/metrics/scrolls.py
        rouge_scores["geometric_mean"] = (
            rouge_scores["rouge1"] * rouge_scores["rouge2"] * rouge_scores["rougeL"]
        ) ** (1.0 / 3.0)
        for k, v in rouge_scores.items():
            scores[f"rouge/{k}"] = round(v * 100, 1)
        rouge_scores = rouge_metric.compute(
            predictions=pred,
            references=tgt,
            use_stemmer=True,
            use_aggregator=False,
        )
        rouge_scores["geometric_mean"] = [
            (r1 * r2 * rl) ** (1.0 / 3.0)
            for r1, r2, rl in zip(
                rouge_scores["rouge1"],
                rouge_scores["rouge2"],
                rouge_scores["rougeL"],
            )
        ]
        for k, v in rouge_scores.items():
            per_ex_scores[f"rouge/{k}"] = [round(_v * 100, 1) for _v in v]

    # AutoACU
    # A3CU
    if "a3cu" in metrics.split(","):
        logger.info("computing a3cu...")
        if isinstance(tgt[0], list):
            # first, check if all examples have the same number of references
            num_tgt = [len(item) for item in tgt]
            if np.mean(num_tgt) != np.max(num_tgt):
                logger.warning("found varying number of references, skipping A3CU")
                return scores, per_ex_scores
            # for each reference, compute scores and take max?
            # multi-ref rouge uses max
            recall_scores, prec_scores, f1_scores = [], [], []
            for i in range(num_tgt[0]):
                ref = [item[i] for item in tgt]
                recall, prec, f1 = a3cu.score(
                    references=ref,
                    candidates=pred,
                    batch_size=16,
                    verbose=False,
                )
                recall_scores += [recall]
                prec_scores += [prec]
                f1_scores += [f1]
            f1_scores = np.array(f1_scores)
            if tgt_pooling == "max":
                best_ref_indices = np.argmax(f1_scores, axis=0)
                recall_scores = [
                    recall_scores[best_ref_idx][ex_idx]
                    for ex_idx, best_ref_idx in enumerate(best_ref_indices)
                ]
                prec_scores = [
                    prec_scores[best_ref_idx][ex_idx]
                    for ex_idx, best_ref_idx in enumerate(best_ref_indices)
                ]
                f1_scores = [
                    f1_scores[best_ref_idx][ex_idx]
                    for ex_idx, best_ref_idx in enumerate(best_ref_indices)
                ]
            elif tgt_pooling == "mean":
                recall_scores = np.mean(recall_scores, axis=0)
                prec_scores = np.mean(prec_scores, axis=0)
                f1_scores = np.mean(f1_scores, axis=0)
            else:
                msg = f"invalid pooling method: {tgt_pooling}"
                raise ValueError(msg)
        else:
            recall_scores, prec_scores, f1_scores = a3cu.score(
                references=tgt,
                candidates=pred,
                batch_size=16,
                verbose=False,
            )
        scores["a3cu/recall"] = round(np.mean(recall_scores) * 100, 1)
        scores["a3cu/precision"] = round(np.mean(prec_scores) * 100, 1)
        scores["a3cu/f1"] = round(np.mean(f1_scores) * 100, 1)
        per_ex_scores["a3cu/recall"] = [round(item * 100, 1) for item in recall_scores]
        per_ex_scores["a3cu/precision"] = [round(item * 100, 1) for item in prec_scores]
        per_ex_scores["a3cu/f1"] = [round(item * 100, 1) for item in f1_scores]

    return scores, per_ex_scores


def compute_summ_metrics(
    pred: list[str] | list[list[str]],
    tgt: list[str] | list[list[str]],
    artifacts_dir: Path,
    tgt_pooling: str = "max",
    metrics: str = "rouge,a3cu",
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """
    Compute summarization metrics.

    Supports,
    1. ROUGE
    2. AutoACU (A3CU)
    """
    # basic pre-processing for rougeLSum computation
    pred, tgt = postprocess_text(preds=pred, labels=tgt)

    # setup A3CU model
    a3cu = None
    if "a3cu" in metrics.split(","):
        a3cu_model_pt = artifacts_dir / "huggingface/model/Yale-LILY/a3cu"
        a3cu = A3CU(model_pt=a3cu_model_pt)

    if isinstance(pred[0], list):
        # if we have multiple predictions per example
        # first, check if all examples have the same number of predictions
        # then compute scores for each set, take avg.
        num_preds = [len(item) for item in pred]
        if np.mean(num_preds) != np.max(num_preds):
            logger.warning("found varying number of predictions, skipping A3CU")
            msg = "varying number of predictions"
            raise ValueError(msg)

        pred = list(zip(*pred))
        scores, per_ex_scores = defaultdict(list), defaultdict(list)
        for pred_set in pred:
            _scores, _per_ex_scores = get_scores(
                pred=list(pred_set),
                tgt=tgt,
                metrics=metrics,
                a3cu=a3cu,
                tgt_pooling=tgt_pooling,
            )
            for k, v in _scores.items():
                scores[k] += [v]
            for k, v in _per_ex_scores.items():
                per_ex_scores[k] += [v]

        # get average scores
        score_stats, per_ex_score_stats = {}, {}
        for k, v in scores.items():
            score_stats[f"{k}/avg"] = round(np.mean(v), 1)
            score_stats[f"{k}/std"] = round(np.std(v), 1)
        for k, v in per_ex_scores.items():
            per_ex_score_stats[f"{k}/avg"] = [
                round(np.mean(item), 1) for item in zip(*v)
            ]
            per_ex_score_stats[f"{k}/std"] = [
                round(np.std(item), 1) for item in zip(*v)
            ]
        return score_stats, per_ex_score_stats

    # single prediction per example
    scores, per_ex_scores = get_scores(
        pred=pred,
        tgt=tgt,
        metrics=metrics,
        a3cu=a3cu,
        tgt_pooling=tgt_pooling,
    )

    return scores, per_ex_scores
