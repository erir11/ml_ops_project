from typing import Any

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)
from torchvision import models


class CarDamageModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        scheduler_step_size: int = 7,
        scheduler_gamma: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the ResNet model with pretrained weights
        if model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unsupported model_name '{model_name}'. Choose 'resnet50' or 'resnet101'.")

        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics - now only F1 score and confusion matrix
        self.f1_score = MulticlassF1Score(
            num_classes=num_classes,
            average='macro',
            zero_division=0.0  # Handle division by zero by returning 0
        )
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(
            optimizer, step_size=self.hparams.scheduler_step_size, gamma=self.hparams.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: Any, batch_idx: int):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        # Only log loss and F1
        f1 = self.f1_score(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        # Log only F1 score
        f1 = self.f1_score(preds, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

        # Collect predictions for confusion matrix
        self.confusion_matrix(preds, labels)

    def on_validation_epoch_end(self):
        # Compute and log confusion matrix
        cm = self.confusion_matrix.compute()
        fig = self.plot_confusion_matrix(cm)

        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                "confusion_matrix": wandb.Image(fig),
                "epoch": self.current_epoch
            })

        plt.close(fig)
        self.confusion_matrix.reset()

    def test_step(self, batch: Any, batch_idx: int):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        # Log only F1 score
        f1 = self.f1_score(preds, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True)

        # Collect predictions for confusion matrix
        self.confusion_matrix(preds, labels)

    def on_test_epoch_end(self):
        # Compute and log confusion matrix
        cm = self.confusion_matrix.compute()
        fig = self.plot_confusion_matrix(cm)

        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                "test_confusion_matrix": wandb.Image(fig),
                "epoch": self.current_epoch
            })

        plt.close(fig)
        self.confusion_matrix.reset()

    def plot_confusion_matrix(self, cm: torch.Tensor) -> plt.Figure:
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        return fig
