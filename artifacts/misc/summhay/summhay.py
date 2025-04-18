"""Dataset script for summary of a haystack dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator

import datasets

# get environ variable
DATA = Path(os.getenv("HF_DATA_PATH", None)) / "misc/summhay"


class SummHayConfig(datasets.BuilderConfig):
    """BuilderConfig for SummHay."""

    def __init__(self: SummHayConfig, features: list, **kwargs: dict) -> None:
        """Init config."""
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.features = features


class SummHay(datasets.GeneratorBasedBuilder):
    """SummHay dataset."""

    BUILDER_CONFIGS: ClassVar[list] = [
        SummHayConfig(
            name="summhay",
            description="SummHay dataset.",
            features=["document", "summary", "query", "id"],
        ),
        SummHayConfig(
            name="summhay_oracle",
            description="SummHay dataset with oracle documents.",
            features=["document", "summary", "query", "id"],
        ),
    ]

    def _info(self: SummHay) -> datasets.DatasetInfo:
        features = {}
        for feature in self.config.features:
            if feature == "document":
                features[feature] = datasets.Sequence(datasets.Value("string"))
            else:
                features[feature] = datasets.Value("string")
        return datasets.DatasetInfo(
            description="summary of a haystack dataset.",
            features=datasets.Features(features),
        )

    def _split_generators(
        self: SummHay,
        dl_manager: datasets.DownloadManager,  # noqa: ARG002
    ) -> list:
        filepaths = [DATA / f"data/topic_conv{i}.json" for i in range(1, 6)]
        filepaths += [DATA / f"data/topic_news{i}.json" for i in range(1, 6)]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepaths": filepaths,
                },
            ),
        ]

    def _generate_examples(
        self: SummHay,
        filepaths: list[str],
    ) -> Iterator[tuple[int, dict]]:
        for filepath in filepaths:
            with filepath.open(encoding="utf-8") as f:
                data = json.load(f)
                for subtopic_item in data["subtopics"]:
                    summary = [
                        insight_item["insight"]
                        for insight_item in subtopic_item["insights"]
                    ]
                    summary = " ".join(summary)  # concatenate all insights
                    query = subtopic_item["query"]
                    subtopic_id = subtopic_item["subtopic_id"]
                    if self.config.name == "summhay":
                        docs = [
                            doc_item["document_text"] for doc_item in data["documents"]
                        ]
                        yield (
                            subtopic_id,
                            {
                                "document": docs,
                                "summary": summary,
                                "query": query,
                                "id": subtopic_id,
                            },
                        )
                    elif self.config.name == "summhay_oracle":
                        insight_ids = {
                            insight_item["insight_id"]
                            for insight_item in subtopic_item["insights"]
                        }
                        oracle_docs = [
                            doc_item["document_text"]
                            for doc_item in data["documents"]
                            if len(set(doc_item["insights_included"]) & insight_ids) > 0
                        ]
                        yield (
                            subtopic_id,
                            {
                                "document": oracle_docs,
                                "summary": summary,
                                "query": query,
                                "id": subtopic_id,
                            },
                        )
