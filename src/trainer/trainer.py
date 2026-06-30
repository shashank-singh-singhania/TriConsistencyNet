"""
TriConsistencyNet

Trainer Engine

Author: Shashank Singh
"""

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None,
        mixed_precision=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        all_probabilities = []

        for batch in tqdm(dataloader, desc="Training Batch", leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += images.size(0)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.detach().cpu().numpy())

        return {
            "loss": running_loss / total,
            "accuracy": correct / total,
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probabilities,
        }

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation Batch", leave=False):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += images.size(0)

                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.detach().cpu().numpy())

        return {
            "loss": running_loss / total,
            "accuracy": correct / total,
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probabilities,
        }
