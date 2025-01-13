from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CarDamageDataset(Dataset):
    """Dataset for car damage classification."""

    def __init__(self, data_dir: Path, split: str = "train", transform=None):
        self.data_dir = data_dir / split
        self.transform = transform

        # Check if the CSV file exists
        csv_path = self.data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Validate the required columns exist
        required_columns = ["filename", "label"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate all image files exist
        images_dir = self.data_dir / "images"
        missing_images = []
        for filename in self.df["filename"]:
            if not (images_dir / filename).exists():
                missing_images.append(filename)
        if missing_images:
            raise FileNotFoundError(f"Missing {len(missing_images)} image files. First few: {missing_images[:5]}")

        # Validate labels
        unique_labels = self.df["label"].unique()
        print(f"Found {len(unique_labels)} unique labels: {sorted(unique_labels)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        try:
            # Load image
            image_path = self.data_dir / "images" / row["filename"]
            image = np.array(Image.open(image_path).convert("RGB"))

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            # Get label
            label = row["label"] - 1  # Convert to 0-based index

            return image, label

        except Exception as e:
            print(f"Error loading image {row['filename']}: {str(e)}")
            raise


class CarDamageDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for car damage classification."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (256, 256),  # Match ResNet expected input
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        # Define transforms
        self.train_transform = A.Compose(
            [
                A.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.8, 1.0),  # Only crop up to 20% of the image
                    ratio=(0.75, 1.333),  # Standard aspect ratio range
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        # Fixed GaussNoise parameters
                        A.GaussNoise(p=0.5),  # Removed 'mean' parameter and fixed var_limit format
                        A.GaussianBlur(p=0.5),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        # Fixed OpticalDistortion parameters
                        A.OpticalDistortion(distort_limit=0.05, p=0.5),  # Removed 'shift_limit' parameter
                        A.GridDistortion(distort_limit=0.1, p=0.5),
                    ],
                    p=0.3,
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        self.val_transform = A.Compose(
            [
                A.Resize(height=self.image_size[0], width=self.image_size[1], interpolation=Image.BICUBIC),
                A.CenterCrop(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = CarDamageDataset(self.data_dir, split="train", transform=self.train_transform)
            self.val_dataset = CarDamageDataset(self.data_dir, split="val", transform=self.val_transform)

            # Print dataset sizes
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = CarDamageDataset(self.data_dir, split="test", transform=self.val_transform)
            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=True,
            drop_last=True,  # Prevent issues with last incomplete batch
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # pin_memory=True,
            persistent_workers=True,
        )


# Example usage:
if __name__ == "__main__":
    # Test the data pipeline
    data_dir = Path("data/processed")
    # Create DataModule
    datamodule = CarDamageDataModule(data_dir)
    datamodule.setup()

    # Get a batch from train loader
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch

    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    # # Save an example image to verify
    # example_image = images[0]
    # example_label = labels[0].item()

    # # Unnormalize the image
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # save_image = example_image.detach().cpu().numpy().transpose(1, 2, 0)
    # save_image = std * save_image + mean
    # save_image = np.clip(save_image, 0, 1)
    # save_image = (save_image * 255).astype(np.uint8)
    # save_image = Image.fromarray(save_image)

    # # Save the image to file
    # save_path = Path("example_image.png")
    # save_image.save(save_path)
    # print(f"Example image saved to {save_path} with label {example_label}")
