from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.splitter import FaceForensicsSplitter


def main():

    splitter = FaceForensicsSplitter()

    train_df, val_df, test_df = splitter.save()

    print()

    print("Train :", len(train_df))
    print("Validation :", len(val_df))
    print("Test :", len(test_df))

    print()

    print("Train Distribution")

    print(
        train_df["Manipulation"]
        .value_counts()
        .sort_index()
    )

    print()

    print("Validation Distribution")

    print(
        val_df["Manipulation"]
        .value_counts()
        .sort_index()
    )

    print()

    print("Test Distribution")

    print(
        test_df["Manipulation"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":

    main()