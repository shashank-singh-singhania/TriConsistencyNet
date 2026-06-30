"""
TriConsistencyNet

Checkpoint Manager

Author: Shashank Singh
"""

from pathlib import Path

import torch

from src.utils.logger import project_logger


class CheckpointManager:

    def __init__(self, checkpoint_directory):

        self.directory = Path(checkpoint_directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model,
        optimizer,
        epoch,
        metric,
        filename="best_model.pth",
    ):

        checkpoint = {

            "epoch": epoch,

            "metric": metric,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),
        }

        output_path = self.directory / filename

        torch.save(
            checkpoint,
            output_path,
        )

        project_logger.success(
            f"Checkpoint saved: {output_path}"
        )

    def load(
        self,
        model,
        optimizer,
        checkpoint_path,
        device,
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        project_logger.success(
            "Checkpoint loaded successfully."
        )

        return (
            checkpoint["epoch"],
            checkpoint["metric"],
        )
