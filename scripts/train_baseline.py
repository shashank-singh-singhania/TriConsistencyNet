"""
TriConsistencyNet

Baseline Training Entrypoint

Author: Shashank Singh
"""

from pathlib import Path
import sys

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import torch.nn as nn

from src.datasets.dataloader import get_dataloaders
from src.models.efficientnet_baseline import EfficientNetBaseline
from src.trainer.checkpoint import CheckpointManager
from src.trainer.metrics import Metrics
from src.trainer.trainer import Trainer
from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a quick verification dry-run")
    args = parser.parse_args()

    project_logger.info("Initializing baseline training...")

    # 1. Load Configurations
    training_config = ConfigLoader().load("training.yaml")
    dataset_config = ConfigLoader().load("dataset.yaml")

    # Resolve paths
    metadata_csv = (
        PROJECT_ROOT
        / "dataset"
        / "FaceForensics++"
        / "metadata"
        / "face_metadata.csv"
    )
    train_split_csv = PROJECT_ROOT / "dataset" / "splits" / "train.csv"

    # 2. Compute Class Weights to handle dataset imbalance
    if metadata_csv.exists():
        metadata_df = pd.read_csv(metadata_csv)
        class_counts = metadata_df["label"].value_counts()
        n_samples = len(metadata_df)
        n_classes = 2

        n_real = class_counts.get("REAL", 1)
        n_fake = class_counts.get("FAKE", 1)

        weight_real = n_samples / (n_classes * n_real)
        weight_fake = n_samples / (n_classes * n_fake)

        class_weights = torch.tensor(
            [weight_real, weight_fake], dtype=torch.float
        )
        project_logger.info(
            f"Class balance - REAL: {n_real}, FAKE: {n_fake}. "
            f"Computed Weights - REAL: {weight_real:.4f}, FAKE: {weight_fake:.4f}"
        )
    else:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)
        project_logger.warning(
            "Metadata CSV not found. Defaulting to equal class weights."
        )

    # 3. Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders()
    project_logger.info(
        f"DataLoaders created - Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    # 4. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_logger.info(f"Using device: {device}")

    # 5. Instantiate Model
    model = EfficientNetBaseline().to(device)

    # 6. Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    lr = training_config.training.learning_rate
    optimizer_name = training_config.training.optimizer
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    epochs = training_config.training.epochs
    scheduler_name = training_config.training.scheduler
    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    else:
        scheduler = None

    # Dry-run override
    if args.dry_run:
        train_loader.dataset.dataframe = train_loader.dataset.dataframe.head(100)
        if len(val_loader.dataset) > 0:
            val_loader.dataset.dataframe = val_loader.dataset.dataframe.head(100)
        epochs = 1
        project_logger.info("Dry-run mode active: dataset sliced to 100 rows, epochs set to 1.")

    # 7. Initialize Trainer and Checkpoint Manager
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        mixed_precision=training_config.training.mixed_precision,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_directory=PROJECT_ROOT / "checkpoints"
    )

    best_val_acc = -1.0

    # 8. Training Loop
    project_logger.info(f"Starting baseline training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        project_logger.info(f"--- Epoch {epoch:02d}/{epochs:02d} ---")

        # Train one epoch
        train_results = trainer.train_epoch(train_loader)
        train_metrics = Metrics.calculate(
            targets=train_results["labels"],
            predictions=train_results["predictions"],
            probabilities=train_results["probabilities"],
        )

        # Validate if validation data is available
        if len(val_loader.dataset) > 0:
            val_results = trainer.validate(val_loader)
            val_metrics = Metrics.calculate(
                targets=val_results["labels"],
                predictions=val_results["predictions"],
                probabilities=val_results["probabilities"],
            )
            val_loss = val_results["loss"]
            val_acc = val_metrics["accuracy"]
        else:
            val_loss = 0.0
            val_acc = 0.0
            val_metrics = {"accuracy": 0.0}

        # Print Epoch Summary
        project_logger.info(
            f"Train Loss: {train_results['loss']:.4f} | Train Acc: {train_metrics['accuracy'] * 100:.2f}%"
        )
        if len(val_loader.dataset) > 0:
            project_logger.info(
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
            )

        # Save Latest Checkpoint
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric=val_acc,
            filename="latest_model.pth",
        )

        # Save Best Checkpoint if validation accuracy improves (or if validating is skipped, save first run)
        if val_acc > best_val_acc or len(val_loader.dataset) == 0:
            best_val_acc = val_acc
            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric=best_val_acc,
                filename="best_model.pth",
            )
            project_logger.success(
                f"Best model updated with metric: {best_val_acc * 100:.2f}%"
            )

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":
    main()
