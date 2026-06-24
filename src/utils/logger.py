"""
TriConsistencyNet

Central Logging Utility

Author: Shashank Singh

Description
-----------
Provides a unified logger instance across the entire project.
"""

from pathlib import Path

from loguru import logger

LOG_DIRECTORY = Path("logs")

LOG_DIRECTORY.mkdir(exist_ok=True)

logger.remove()

logger.add(
    LOG_DIRECTORY / "project.log",
    rotation="10 MB",
    retention="14 days",
    level="INFO",
    enqueue=True,
)

logger.add(
    sink=lambda msg: print(msg, end=""),
    format="{level}\n{message}",
    level="INFO",
)

project_logger = logger