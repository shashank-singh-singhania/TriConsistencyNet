from pathlib import Path
import sys
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.statistics import DatasetStatistics


def main():

    csv_path = (
        PROJECT_ROOT
        / "dataset"
        / "splits"
        / "train.csv"
    )

    output_path = (
        PROJECT_ROOT
        / "reports"
        / "dataset_statistics.json"
    )

    statistics = DatasetStatistics(csv_path)

    result = statistics.save(output_path)

    pprint(result)


if __name__ == "__main__":

    main()