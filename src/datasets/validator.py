"""
TriConsistencyNet

FaceForensics++ Dataset Validator

Author: Shashank Singh
"""

from pathlib import Path

import pandas as pd

from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


REQUIRED_COLUMNS = [
    "File Path",
    "Label",
    "Frame Count",
    "Width",
    "Height",
    "Codec",
    "File Size(MB)",
]


class FaceForensicsValidator:

    def __init__(self, dataset_root=None, metadata_csv=None):

        config = ConfigLoader().load("dataset.yaml")

        self.dataset_root = (
            Path(dataset_root)
            if dataset_root is not None
            else Path(config.dataset.root)
        )

        self.metadata_path = (
            Path(metadata_csv)
            if metadata_csv is not None
            else Path(config.dataset.metadata)
        )

        self.dataframe = None

    def load_metadata(self):

        project_logger.info("Loading FaceForensics++ metadata...")

        self.dataframe = pd.read_csv(self.metadata_path)

        return self.dataframe

    def validate_columns(self):

        missing_columns = []

        for column in REQUIRED_COLUMNS:

            if column not in self.dataframe.columns:

                missing_columns.append(column)

        if missing_columns:

            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        project_logger.success("Column validation passed.")

    def validate_labels(self):

        labels = set(self.dataframe["Label"].unique())

        allowed = {"REAL", "FAKE"}

        invalid = labels - allowed

        if invalid:

            raise ValueError(
                f"Invalid labels: {invalid}"
            )

        project_logger.success("Label validation passed.")

    def validate_paths(self):

        missing = []

        for relative_path in self.dataframe["File Path"]:

            absolute_path = self.dataset_root / relative_path

            if not absolute_path.exists():

                missing.append(str(relative_path))

        if missing:

            project_logger.warning(
                f"{len(missing)} missing files detected."
            )

        else:

            project_logger.success(
                "All video files exist."
            )

        return missing

    def generate_statistics(self):

        stats = {

            "total_videos": len(self.dataframe),

            "real_videos": int(
                (self.dataframe["Label"] == "REAL").sum()
            ),

            "fake_videos": int(
                (self.dataframe["Label"] == "FAKE").sum()
            ),

            "average_frames": round(
                float(self.dataframe["Frame Count"].mean()),
                2,
            ),

            "average_width": round(
                float(self.dataframe["Width"].mean()),
                2,
            ),

            "average_height": round(
                float(self.dataframe["Height"].mean()),
                2,
            ),

            "average_size_mb": round(
                float(self.dataframe["File Size(MB)"].mean()),
                2,
            ),
        }

        return stats

    def run(self):

        self.load_metadata()

        self.validate_columns()

        self.validate_labels()

        missing = self.validate_paths()

        statistics = self.generate_statistics()

        return {

            "missing_files": missing,

            "statistics": statistics,

            "dataframe": self.dataframe,

        }