"""
Production Frame Extraction Script

Usage:

python scripts/extract_frames.py --split train
python scripts/extract_frames.py --split val
python scripts/extract_frames.py --split test
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.frame_extractor import FrameExtractor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
    )

    args = parser.parse_args()

    split_csv = (
        PROJECT_ROOT
        / "dataset"
        / "splits"
        / f"{args.split}.csv"
    )

    extractor = FrameExtractor()

    extractor.extract_from_split(split_csv)


if __name__ == "__main__":

    main()