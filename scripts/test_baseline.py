from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.efficientnet_baseline import (
    EfficientNetBaseline,
)


def main():

    model = EfficientNetBaseline()

    dummy = torch.randn(
        4,
        3,
        224,
        224,
    )

    output = model(dummy)

    print()

    print("Input :", dummy.shape)

    print("Output:", output.shape)


if __name__ == "__main__":

    main()