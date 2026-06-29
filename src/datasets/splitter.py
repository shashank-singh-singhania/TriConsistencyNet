"""
TriConsistencyNet

FaceForensics++ Split Generator

Author: Shashank Singh
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


class FaceForensicsSplitter:

    def __init__(self):

        config = ConfigLoader().load("dataset.yaml")

        self.metadata_path = Path(config.dataset.metadata)

        self.train_ratio = config.dataset.train_ratio
        self.val_ratio = config.dataset.val_ratio
        self.test_ratio = config.dataset.test_ratio

        self.df = pd.read_csv(self.metadata_path)

        self._prepare_dataframe()

    def _prepare_dataframe(self):

        """
        Add manipulation information and stratification key.
        """

        self.df["Manipulation"] = self.df["File Path"].apply(
            lambda x: Path(x).parts[0]
        )

        self.df["StratifyKey"] = (
            self.df["Label"]
            + "_"
            + self.df["Manipulation"]
        )

    def split(self):

        train_df, temp_df = train_test_split(
            self.df,
            test_size=(1 - self.train_ratio),
            random_state=42,
            shuffle=True,
            stratify=self.df["StratifyKey"],
        )

        test_ratio_adjusted = (
            self.test_ratio
            / (self.val_ratio + self.test_ratio)
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio_adjusted,
            random_state=42,
            shuffle=True,
            stratify=temp_df["StratifyKey"],
        )

        return train_df, val_df, test_df

    def save(self):

        train_df, val_df, test_df = self.split()

        output_dir = (
            Path(__file__).resolve().parents[2]
            / "dataset"
            / "splits"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_df.to_csv(
            output_dir / "train.csv",
            index=False,
        )

        val_df.to_csv(
            output_dir / "val.csv",
            index=False,
        )

        test_df.to_csv(
            output_dir / "test.csv",
            index=False,
        )

        project_logger.success(
            "Dataset splits generated successfully."
        )

        return train_df, val_df, test_df