"""
TriConsistencyNet

Face Metadata Generator

Author: Shashank Singh
"""

from pathlib import Path

import pandas as pd

from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


class FaceMetadataGenerator:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]

        config = ConfigLoader().load("dataset.yaml")

        self.faces_root = self.project_root / Path(config.dataset.processed) / "faces"

        self.output_directory = self.project_root / "dataset" / "FaceForensics++" / "metadata"

    def generate(self):

        rows = []

        for manipulation_dir in sorted(self.faces_root.iterdir()):

            if not manipulation_dir.is_dir():

                continue

            manipulation = manipulation_dir.name

            label = (
                "REAL"
                if manipulation == "original"
                else "FAKE"
            )

            for video_dir in sorted(manipulation_dir.iterdir()):

                if not video_dir.is_dir():

                    continue

                video_id = video_dir.name

                for image_path in sorted(video_dir.glob("*.png")):

                    rows.append(

                        {

                            "face_path":

                            str(

                                image_path.resolve().relative_to(

                                    self.project_root

                                )

                            ).replace("\\", "/"),

                            "label": label,

                            "manipulation": manipulation,

                            "video_id": video_id,

                            "frame_name": image_path.stem,

                        }

                    )

        dataframe = pd.DataFrame(rows)

        if not dataframe.empty:

            dataframe = dataframe[

                [

                    "face_path",

                    "label",

                    "manipulation",

                    "video_id",

                    "frame_name",

                ]

            ]

        self.output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        output_path = self.output_directory / "face_metadata.csv"

        try:

            dataframe.to_csv(

                output_path,

                index=False,

            )

        except PermissionError:

            fallback_path = self.output_directory / "face_metadata_generated.csv"

            dataframe.to_csv(

                fallback_path,

                index=False,

            )

            project_logger.warning(

                "face_metadata.csv is in use; wrote face_metadata_generated.csv instead."

            )

        project_logger.success(

            f"Generated {len(dataframe)} entries."

        )

        return dataframe