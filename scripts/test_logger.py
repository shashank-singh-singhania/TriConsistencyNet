from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import project_logger


def main():

    project_logger.info("Logger initialized successfully.")

    project_logger.warning("This is a warning message.")

    project_logger.success("Everything is working correctly.")

    project_logger.error("Example error message.")


if __name__ == "__main__":

    main()