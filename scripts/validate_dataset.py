from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.validator import FaceForensicsValidator


def main():
    validator = FaceForensicsValidator(
        dataset_root=PROJECT_ROOT / "dataset" / "FaceForensics++" / "videos",
        metadata_csv=PROJECT_ROOT / "dataset" / "FaceForensics++" / "videos" / "csv" / "FF++_Metadata.csv",
    )

    result = validator.run()

    print(result["statistics"])
    print("Missing Files:", len(result["missing_files"]))


if __name__ == "__main__":
    main()