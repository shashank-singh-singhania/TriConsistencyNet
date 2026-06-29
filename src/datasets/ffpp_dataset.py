"""
TriConsistencyNet

FaceForensics++ PyTorch Dataset

Author: Shashank Singh
"""

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import ConfigLoader


class FFPPDataset(Dataset):

    def __init__(
        self,
        metadata_csv,
        transform=None,
    ):

        self.dataframe = pd.read_csv(metadata_csv)

        self.transform = transform

        config = ConfigLoader().load("dataset.yaml")

        self.image_size = config.dataset.image_size

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = Path(row["face_path"])

        image = cv2.imread(str(image_path))

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

            "label": torch.tensor(label),

            "manipulation": row["manipulation"],

            "video_id": row["video_id"],

            "frame_name": str(row["frame_name"]).zfill(6),
        }