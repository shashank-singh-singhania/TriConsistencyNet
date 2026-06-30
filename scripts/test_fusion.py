from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.triconsistencynet.fusion import (
    AdaptiveFeatureFusion,
)


def main():

    model = AdaptiveFeatureFusion()

    x = torch.randn(
        4,
        1280,
        7,
        7,
    )

    y = model(x)

    print()

    print("Input :", x.shape)

    print("Output:", y.shape)


if __name__ == "__main__":

    main()
