from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.frame_extractor import FrameExtractor


def main():

    train_csv = (

        PROJECT_ROOT

        / "dataset"

        / "splits"

        / "train.csv"

    )

    dataframe = pd.read_csv(train_csv)

    extractor = FrameExtractor()

    extractor.extract_from_dataframe(

        dataframe,

        limit=1,

    )


if __name__ == "__main__":

    main()