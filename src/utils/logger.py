"""
TriConsistencyNet

Central Logging Utility

Author: Shashank Singh

Description
-----------
Provides a unified logger instance for the complete project.
"""

from pathlib import Path

from loguru import logger


LOG_DIRECTORY = Path("logs")
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(
    LOG_DIRECTORY / "project.log",
    rotation="10 MB",
    retention="14 days",
    level="INFO",
    enqueue=True,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level:<8} | "
        "{name}:{function}:{line} | "
        "{message}"
    ),
)

logger.add(
    sink=lambda message: print(message, end=""),
    format="{level:<8} {message}",
    level="INFO",
)

project_logger = logger