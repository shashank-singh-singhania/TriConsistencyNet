"""
TriConsistencyNet

DataLoader Factory

Author: Shashank Singh
"""

from pathlib import Path
from torch.utils.data import DataLoader
from src.datasets.ffpp_dataset import FFPPDataset
from src.utils.config import ConfigLoader


def get_dataloaders(
    metadata_csv=None,
    train_split_csv=None,
    val_split_csv=None,
    test_split_csv=None,
    batch_size=None,
    num_workers=None,
    pin_memory=None,
    train_transform=None,
    val_transform=None,
):
    """
    DataLoader factory to instantiate and return Train, Validation, and Test data loaders.
    """
    # Load default configurations
    training_config = ConfigLoader().load("training.yaml")

    # Resolve project root
    project_root = Path(__file__).resolve().parents[2]

    # Resolve paths
    if metadata_csv is None:
        metadata_csv = (
            project_root
            / "dataset"
            / "FaceForensics++"
            / "metadata"
            / "face_metadata.csv"
        )
    else:
        metadata_csv = Path(metadata_csv)

    if train_split_csv is None:
        train_split_csv = project_root / "dataset" / "splits" / "train.csv"
    else:
        train_split_csv = Path(train_split_csv)

    if val_split_csv is None:
        val_split_csv = project_root / "dataset" / "splits" / "val.csv"
    else:
        val_split_csv = Path(val_split_csv)

    if test_split_csv is None:
        test_split_csv = project_root / "dataset" / "splits" / "test.csv"
    else:
        test_split_csv = Path(test_split_csv)

    # Resolve training parameters
    if batch_size is None:
        batch_size = training_config.training.batch_size

    if num_workers is None:
        num_workers = training_config.training.num_workers

    if pin_memory is None:
        pin_memory = training_config.training.pin_memory

    # Instantiate datasets
    train_dataset = FFPPDataset(
        metadata_csv=metadata_csv,
        split_csv=train_split_csv,
        transform=train_transform,
    )

    val_dataset = FFPPDataset(
        metadata_csv=metadata_csv,
        split_csv=val_split_csv,
        transform=val_transform,
    )

    test_dataset = FFPPDataset(
        metadata_csv=metadata_csv,
        split_csv=test_split_csv,
        transform=val_transform,
    )

    # Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
