from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataloader import get_dataloaders


def main():

    print("Initializing dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders()

    print()
    print("Train Loader Size:", len(train_loader))
    print("Val Loader Size:", len(val_loader))
    print("Test Loader Size:", len(test_loader))

    # Retrieve one batch from the train loader
    for batch in train_loader:
        print()
        print("Successfully loaded batch!")
        print("Image Batch Shape:", batch["image"].shape)
        print("Label Batch Shape:", batch["label"].shape)
        print("First Label in Batch:", batch["label"][0].item())
        break


if __name__ == "__main__":

    main()
