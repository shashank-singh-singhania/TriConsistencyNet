from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.triconsistencynet import (
    CrossConsistencyAttention,
)


def main():

    model = CrossConsistencyAttention()

    spatial = torch.randn(
        4,
        1280,
        7,
        7,
    )

    frequency = torch.randn(
        4,
        256,
        7,
        7,
    )

    refined, attention = model(
        spatial,
        frequency,
    )

    print()

    print("Spatial :", spatial.shape)

    print("Frequency :", frequency.shape)

    print()

    print("Refined :", refined.shape)

    print("Attention :", attention.shape)


if __name__ == "__main__":

    main()
