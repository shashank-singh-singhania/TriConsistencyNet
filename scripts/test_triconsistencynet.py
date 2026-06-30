from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.triconsistencynet import TriConsistencyNet


def main():

    model = TriConsistencyNet()

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    y = model(x)

    print()

    print("Input :", x.shape)

    print("Output:", y.shape)

    print()

    print(
        "Attention:",
        model.last_attention_map.shape,
    )


if __name__ == "__main__":

    main()
