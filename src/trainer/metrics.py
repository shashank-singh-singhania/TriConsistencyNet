"""
TriConsistencyNet

Evaluation Metrics

Author: Shashank Singh
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class Metrics:

    @staticmethod
    def calculate(
        targets,
        predictions,
        probabilities=None,
    ) -> Dict:

        targets = np.asarray(targets)
        predictions = np.asarray(predictions)

        results = {

            "accuracy": accuracy_score(
                targets,
                predictions,
            ),

            "precision": precision_score(
                targets,
                predictions,
                zero_division=0,
            ),

            "recall": recall_score(
                targets,
                predictions,
                zero_division=0,
            ),

            "f1": f1_score(
                targets,
                predictions,
                zero_division=0,
            ),

            "confusion_matrix": confusion_matrix(
                targets,
                predictions,
            ),
        }

        if probabilities is not None:

            probabilities = np.asarray(probabilities)

            results["roc_auc"] = roc_auc_score(
                targets,
                probabilities,
            )

        return results
