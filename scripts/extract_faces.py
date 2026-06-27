"""
Production Face Extraction Script
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.face_extractor import FaceExtractor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes",
    )

    args = parser.parse_args()

    # Diagnostics to verify GPU visibility in environment
    try:
        import torch
        print("PyTorch GPU Available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("PyTorch GPU Name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("PyTorch diagnostic failed:", e)

    try:
        import tensorflow as tf
        print("TensorFlow GPU List:", tf.config.list_physical_devices('GPU'))
    except Exception as e:
        print("TensorFlow diagnostic failed:", e)

    split_csv = (
        PROJECT_ROOT
        / "dataset"
        / "splits"
        / f"{args.split}.csv"
    )

    extractor = FaceExtractor()

    extractor.extract_from_split(split_csv, workers=args.workers)


if __name__ == "__main__":
    main()
