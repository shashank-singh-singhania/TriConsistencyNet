"""
TriConsistencyNet

FaceForensics++ PyTorch Dataset

Author: Shashank Singh
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


class FFPPDataset(Dataset):

    def __init__(
        self,
        metadata_csv,
        split_csv,
        transform=None,
    ):

        self.transform = transform

        config = ConfigLoader().load("dataset.yaml")

        self.image_size = config.dataset.image_size

        metadata_df = pd.read_csv(metadata_csv)

        split_df = pd.read_csv(split_csv)

        # Use a combined (video_id, manipulation) key to prevent cross-manipulation subject leakage
        split_df["video_id"] = split_df["File Path"].apply(lambda x: Path(x).stem).astype(str)
        allowed_keys = set(
            split_df["video_id"] + "_" + split_df["Manipulation"].astype(str)
        )

        metadata_keys = metadata_df["video_id"].astype(str) + "_" + metadata_df["manipulation"].astype(str)

        self.dataframe = metadata_df[
            metadata_keys.isin(allowed_keys)
        ].reset_index(drop=True)

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # Resolve the image path as absolute relative to the project root
        project_root = Path(__file__).resolve().parents[3]
        image_path = project_root / row["face_path"]

        image = cv2.imread(str(image_path))

        if image is None:
            # Gracefully handle missing files during local development/testing on laptop
            image = np.zeros(
                (self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        if self.transform is not None:

            image = self.transform(image=image)["image"]

        else:

            image = torch.from_numpy(image)

            image = image.permute(2, 0, 1)

            image = image.float() / 255.0

        label = 1 if row["label"] == "FAKE" else 0

        return {

            "image": image,

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),

            "image_path": str(image_path),

            "manipulation": row["manipulation"],

            "video_id": row["video_id"],

            "frame_name": str(row["frame_name"]).zfill(6),
        }