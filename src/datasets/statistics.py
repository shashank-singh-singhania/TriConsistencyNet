"""
TriConsistencyNet

Dataset Statistics Generator

Author: Shashank Singh
"""

from pathlib import Path
import json

import pandas as pd

from src.utils.logger import project_logger


class DatasetStatistics:

    def __init__(self, csv_path):

        self.csv_path = Path(csv_path)

        self.df = pd.read_csv(self.csv_path)

    def generate(self):

        statistics = {

            "total_videos": int(len(self.df)),

            "real_videos": int(
                (self.df["Label"] == "REAL").sum()
            ),

            "fake_videos": int(
                (self.df["Label"] == "FAKE").sum()
            ),

            "manipulation_distribution":

                self.df["Manipulation"]
                .value_counts()
                .sort_index()
                .to_dict(),

            "average_frames":

                round(
                    self.df["Frame Count"].mean(),
                    2
                ),

            "average_width":

                round(
                    self.df["Width"].mean(),
                    2
                ),

            "average_height":

                float(
    round(
        self.df["Height"].mean(),
        2,
    )
),

            "average_size_mb":

                round(
                    self.df["File Size(MB)"].mean(),
                    2
                ),
        }

        return statistics

    def save(self, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        statistics = self.generate()

        with open(output_path, "w") as file:

            json.dump(
                statistics,
                file,
                indent=4,
            )

        project_logger.success(
            f"Statistics saved to {output_path}"
        )

        return statistics