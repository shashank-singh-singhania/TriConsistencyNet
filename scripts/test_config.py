from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


def main():

    # Keep this smoke test output focused on config values only.
    project_logger.remove()

    loader = ConfigLoader()

    config = loader.load_all()

    print()

    print("Project Name:")

    print(config["project"].project_name)

    print()

    print("Dataset Root:")

    print(config["dataset"].dataset.root)

    print()

    print("Learning Rate:")

    print(config["training"].training.learning_rate)

    print()

    print("Backbone:")

    print(config["model"].model.backbone)


if __name__ == "__main__":

    main()