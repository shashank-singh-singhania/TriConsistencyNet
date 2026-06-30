"""
TriConsistencyNet

Baseline Model Evaluation Script

Author: Shashank Singh
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataloader import get_dataloaders
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.trainer.metrics import Metrics
from src.trainer.trainer import Trainer
from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


def main():
    project_logger.info("Initializing baseline model evaluation on the test split...")

    # 1. Setup Directories
    experiment_dir = PROJECT_ROOT / "experiments" / "baseline"
    checkpoint_path = experiment_dir / "checkpoints" / "best_model.pth"
    metrics_dir = experiment_dir / "metrics"

    if not checkpoint_path.exists():
        project_logger.error(f"Best checkpoint not found at: {checkpoint_path}")
        return

    # 2. Load Configurations
    training_config = ConfigLoader().load("training.yaml")

    # 3. Create DataLoaders
    _, _, test_loader = get_dataloaders()
    project_logger.info(f"Loaded test split with {len(test_loader.dataset)} face samples ({len(test_loader)} batches).")

    if len(test_loader.dataset) == 0:
        project_logger.error("Test dataset is empty. Ensure you preprocessed the test split.")
        return

    # 4. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_logger.info(f"Using device: {device}")

    # 5. Instantiate Model and Load Checkpoint
    model = EfficientNetBaseline().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    project_logger.success(f"Successfully loaded best checkpoint from Epoch {checkpoint['epoch']} (Val Acc: {checkpoint['metric'] * 100:.2f}%)")

    # 6. Instantiate Trainer for Evaluation
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=None,
        criterion=criterion,
        device=device,
        mixed_precision=training_config.training.mixed_precision,
    )

    # 7. Run Test Evaluation
    project_logger.info("Running inference on test split...")
    test_results = trainer.validate(test_loader)
    
    # 8. Calculate Metrics
    test_metrics = Metrics.calculate(
        targets=test_results["labels"],
        predictions=test_results["predictions"],
        probabilities=test_results["probabilities"],
    )

    # Print Results
    project_logger.success("=== Test Split Metrics ===")
    project_logger.info(f"Test Loss: {test_results['loss']:.4f}")
    project_logger.info(f"Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    project_logger.info(f"Precision: {test_metrics['precision'] * 100:.2f}%")
    project_logger.info(f"Recall:    {test_metrics['recall'] * 100:.2f}%")
    project_logger.info(f"F1-score:  {test_metrics['f1'] * 100:.2f}%")
    project_logger.info(f"ROC-AUC:   {test_metrics['roc_auc'] * 100:.4f}")
    project_logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # 9. Save Metrics to JSON (convert numpy array to list)
    test_metrics_json = test_metrics.copy()
    test_metrics_json["confusion_matrix"] = test_metrics_json["confusion_matrix"].tolist()
    test_metrics_json["loss"] = test_results["loss"]
    
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics_json, f, indent=4)
    project_logger.success(f"Test metrics saved to: {metrics_dir / 'test_metrics.json'}")

    # 10. Generate and Save ROC / PR Curves
    labels = test_results["labels"]
    import numpy as np
    probs = np.asarray(test_results["probabilities"])

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Baseline')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(metrics_dir / "roc_curve.png", dpi=300)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Baseline')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(metrics_dir / "pr_curve.png", dpi=300)
    plt.close()
    
    project_logger.success(f"ROC and PR curves saved to: {metrics_dir}")


if __name__ == "__main__":
    main()
