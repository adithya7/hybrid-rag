"""Download models and datasets on huggingface."""

import sys
from pathlib import Path

import fire
from huggingface_hub import snapshot_download
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
)


def download(repo_id: str, local_dir: str, repo_type: str) -> None:
    """Download from huggingface."""
    local_dir = Path(local_dir) / repo_type / repo_id
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {repo_id} to {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type=repo_type)


if __name__ == "__main__":
    fire.Fire(download)
