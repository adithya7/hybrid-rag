"""GPU related utils."""

from loguru import logger
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
)


def log_gpu_utilization() -> None:
    """Log GPU memory using pynvml."""
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU {} memory use: {:.1f}/{:.1f} GB.".format(
                i, info.used / 1024**3, info.total / 1024**3
            )
        )
