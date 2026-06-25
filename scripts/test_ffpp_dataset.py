from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.ffpp_dataset import FFPPDataset


def main():

    metadata = (
        PROJECT_ROOT
        / "dataset"
        / "FaceForensics++"
        / "metadata"
        / "face_metadata.csv"
    )

    dataset = FFPPDataset(metadata)

    print()

    print("Dataset Size:", len(dataset))

    sample = dataset[0]

    print()

    print("Image Shape:", sample["image"].shape)

    print("Label:", sample["label"])

    print("Manipulation:", sample["manipulation"])

    print("Video:", sample["video_id"])

    print("Frame:", sample["frame_name"])


if __name__ == "__main__":

    main()