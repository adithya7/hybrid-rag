"""Get context window estimates based for a given model and retriever."""

from __future__ import annotations

import sys
from pathlib import Path

import fire
import pandas as pd
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<m>{time:YYYY-MM-DD at HH:mm:ss}</m> | {level} | {message}",
)


def write_bash(data: pd.DataFrame, dataset: str, split: str, output_path: Path) -> None:
    """Write bash script to run the experiments."""
    with output_path.open("w") as wf:
        for _, row in data.iterrows():
            model = row["Model"]
            retriever = row["Retriever"]
            context = row["Context"]
            wf.write(
                f"DATASET={dataset}; SPLIT={split}; M={model}; C={context}; "
                f"bash bash_scripts/sum.sh "
                f"${{M}}_Flat_${{C}}_{retriever} ${{DATASET}} ${{SPLIT}}\n"
            )


def write_scores(
    data: pd.DataFrame, dataset: str, split: str, output_path: Path
) -> None:
    """Write scores for the best estimates."""
    with output_path.open("w") as wf:
        for _, row in data.iterrows():
            model = row["Model"]
            retriever = row["Retriever"]
            context = row["Context"]
            wf.write(
                f"DATASET={dataset}; SPLIT={split}; M={model}; C={context}; "
                f'awk \'{{print $1","$6","$7}}\' <(tail -n1 '
                f"${{DATASET}}_${{M}}_Flat_${{C}}_{retriever}_${{SPLIT}}_scores.txt)\n"
            )


def find_estimate(file_path: str, full_dataset: str | None = None) -> None:
    """Get optimal context window estimates."""
    file_path = Path(file_path)
    dataset = file_path.stem.split("_")[0]
    split = file_path.stem.split("_")[-2]
    data = pd.read_csv(file_path)
    data = data.rename(columns={"Unnamed: 0": "System"})
    # example System name : Qwen25_0p5B_Flat_16K_GTE_1p5B (Model_Flat_Context_GTE_1p5B)
    data["Model"] = data["System"].apply(lambda x: x.split("_Flat_", 1)[0])
    data["Retriever"] = data["System"].apply(lambda x: "_".join(x.rsplit("_", 3)[-2:]))
    data["Context"] = data["System"].apply(lambda x: x.split("_")[-3])
    # remove System column
    del data["System"]
    # reorder columns to have Model, Retriever, Context, a3cu/f1/avg, a3cu/f1/std
    data = data[["Model", "Retriever", "Context", "a3cu/f1/avg", "a3cu/f1/std"]]
    # for a given Model and Retriever, sort by a3cu/f1/avg column
    # create a new dataframe that has the best a3cu/f1/avg for each Model and Retriever

    best_estimates = []
    for retriever in data["Retriever"].unique():
        for model in data["Model"].unique():
            data_sub = data[(data["Model"] == model) & (data["Retriever"] == retriever)]
            data_sub = data_sub.sort_values(
                by=["a3cu/f1/avg", "Context"],
                ascending=[False, True],
            )
            best_estimates.append(data_sub.iloc[0])
    best_estimates = pd.DataFrame(best_estimates)
    if full_dataset:
        output_path = (
            file_path.parent / f"{file_path.stem}_most_performant_{dataset}_{split}.sh"
        )
        write_scores(best_estimates, full_dataset, split, output_path)
    else:
        output_path = file_path.parent / (file_path.stem + "_most_performant.tsv")
        best_estimates.to_csv(output_path, index=False, sep="\t")
        output_path = file_path.parent / (file_path.stem + "_most_performant.sh")
        write_bash(best_estimates, dataset, split, output_path)

    # get best estimates using the following method
    # for a given Model and Retriever, sort by a3cu/f1/avg column
    # pick row with least Context, within the std (a3cu/f1/std) of best a3cu/f1/avg
    # if there are multiple rows with the same a3cu/f1/avg, pick one with least Context
    best_estimates = []
    for retriever in data["Retriever"].unique():
        for model in data["Model"].unique():
            data_sub = data[(data["Model"] == model) & (data["Retriever"] == retriever)]
            data_sub = data_sub.sort_values(
                by=["a3cu/f1/avg", "Context"],
                ascending=[False, True],
            )
            best_a3cu_f1_avg = data_sub.iloc[0]["a3cu/f1/avg"]
            best_a3cu_f1_std = data_sub.iloc[0]["a3cu/f1/std"]
            epsilon = 1e-5  # tolerance for precision issues
            data_sub = data_sub[
                (
                    data_sub["a3cu/f1/avg"]
                    >= (best_a3cu_f1_avg - best_a3cu_f1_std - epsilon)
                )
            ]
            # sort by Context, by converting to int after stripping "K"
            data_sub["Context_int"] = (
                data_sub["Context"].str.replace("K", "").astype(int)
            )
            data_sub = data_sub.sort_values(by="Context_int")
            del data_sub["Context_int"]
            best_estimates.append(data_sub.iloc[0])
    best_estimates = pd.DataFrame(best_estimates)
    if full_dataset:
        output_path = (
            file_path.parent / f"{file_path.stem}_most_efficient_{dataset}_{split}.sh"
        )
        write_scores(best_estimates, full_dataset, split, output_path)
    else:
        output_path = file_path.parent / (file_path.stem + "_most_efficient.tsv")
        best_estimates.to_csv(output_path, index=False, sep="\t")
        output_path = file_path.parent / (file_path.stem + "_most_efficient.sh")
        write_bash(best_estimates, dataset, split, output_path)


if __name__ == "__main__":
    fire.Fire(find_estimate)
