"""
TriConsistencyNet

EfficientNetV2-S Baseline
"""

import timm
import torch.nn as nn

from src.utils.config import ConfigLoader


class EfficientNetBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        config = ConfigLoader().load("model.yaml")

        self.backbone = timm.create_model(
            config.model.backbone,
            pretrained=config.model.pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.dropout = nn.Dropout(config.model.dropout)

        self.classifier = nn.Linear(
            self.backbone.num_features,
            config.model.num_classes,
        )

    def forward(self, x):

        features = self.backbone(x)

        features = self.dropout(features)

        logits = self.classifier(features)

        return logits