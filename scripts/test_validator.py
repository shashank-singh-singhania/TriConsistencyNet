from pathlib import Path
import sys
from pprint import pprint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.validator import FaceForensicsValidator
from src.utils.logger import project_logger


def main():

    project_logger.remove()
    project_logger.add(
        sink=lambda message: print(message, end=""),
        format="{level:<8} {message}",
        level="INFO",
        filter=lambda record: record["message"] != "Loading config: dataset.yaml",
    )

    validator = FaceForensicsValidator()

    result = validator.run()

    print()

    pprint(result["statistics"])

    print()

    print("Missing Files:", len(result["missing_files"]))


if __name__ == "__main__":

    main()