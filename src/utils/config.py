"""
TriConsistencyNet

Configuration Loader

Author: Shashank Singh
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.logger import project_logger


class ConfigLoader:

    def __init__(self, config_directory: str = "configs"):

        self.config_directory = Path(config_directory)

    def load(self, filename: str) -> DictConfig:

        config_path = self.config_directory / filename

        if not config_path.exists():

            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        project_logger.info(f"Loading config: {filename}")

        return OmegaConf.load(config_path)

    def load_all(self):

        project = self.load("project.yaml")

        dataset = self.load("dataset.yaml")

        training = self.load("training.yaml")

        model = self.load("model.yaml")

        return {

            "project": project,

            "dataset": dataset,

            "training": training,

            "model": model,

        }