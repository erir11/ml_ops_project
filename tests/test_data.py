from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ml_ops_project.data import CarDamageDataModule, CarDamageDataset  # Adjust import path as needed


# Helper function to create dummy images
def create_dummy_image(path: Path, size=(256, 256), color=(255, 0, 0)):
    image = Image.new("RGB", size, color)
    image.save(path)


@pytest.fixture
def dummy_data(tmp_path):
    """
    Create a dummy dataset directory structure with CSV files and images.
    """
    data_dir = tmp_path / "processed"
    splits = ["train", "val", "test"]
    num_images = {"train": 10, "val": 5, "test": 5}
    num_classes = 6

    for split in splits:
        split_dir = data_dir / split
        images_dir = split_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create CSV file
        csv_path = split_dir / f"{split}.csv"
        data = {
            "filename": [f"image_{i}.jpg" for i in range(num_images[split])],
            "label": [np.random.randint(1, num_classes + 1) for _ in range(num_images[split])],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        # Create dummy images
        for filename in data["filename"]:
            image_path = images_dir / filename
            create_dummy_image(image_path)

    return data_dir


@pytest.fixture
def corrupted_data(tmp_path):
    """
    Create a dataset with a corrupted image.
    """
    data_dir = tmp_path / "processed_corrupted"
    split = "train"
    images_dir = data_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create CSV file
    csv_path = data_dir / split / f"{split}.csv"
    data = {"filename": ["good_image.jpg", "corrupted_image.jpg"], "label": [1, 2]}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Create a good image
    create_dummy_image(images_dir / "good_image.jpg")

    # Create a corrupted image (write random bytes)
    with open(images_dir / "corrupted_image.jpg", "wb") as f:
        f.write(b"not an image")

    return data_dir


@pytest.fixture
def incomplete_data(tmp_path):
    """
    Create a dataset missing required columns in CSV.
    """
    data_dir = tmp_path / "processed_incomplete"
    split = "train"
    images_dir = data_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create CSV file missing 'label' column
    csv_path = data_dir / split / f"{split}.csv"
    data = {
        "filename": ["image_1.jpg", "image_2.jpg"]
        # 'label' column is missing
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Create dummy images
    for filename in data["filename"]:
        image_path = images_dir / filename
        create_dummy_image(image_path)

    return data_dir


@pytest.fixture
def missing_csv_data(tmp_path):
    """
    Create a dataset directory without the CSV file.
    """
    data_dir = tmp_path / "processed_missing_csv"
    split = "train"
    images_dir = data_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy images without CSV
    image_path = images_dir / "image_1.jpg"
    create_dummy_image(image_path)

    return data_dir


@pytest.fixture
def missing_images_data(tmp_path):
    """
    Create a dataset where some images referenced in CSV are missing.
    """
    data_dir = tmp_path / "processed_missing_images"
    split = "train"
    images_dir = data_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create CSV file referencing two images, one missing
    csv_path = data_dir / split / f"{split}.csv"
    data = {"filename": ["image_1.jpg", "image_missing.jpg"], "label": [1, 2]}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Create only one image
    create_dummy_image(images_dir / "image_1.jpg")

    return data_dir


def test_car_damage_dataset_initialization_success(dummy_data):
    """
    Test successful initialization of CarDamageDataset.
    """
    dataset = CarDamageDataset(data_dir=dummy_data, split="train", transform=None)
    assert len(dataset) == 10
    assert isinstance(dataset, torch.utils.data.Dataset)


def test_car_damage_dataset_missing_csv(missing_csv_data):
    """
    Test initialization failure when CSV file is missing.
    """
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        CarDamageDataset(data_dir=missing_csv_data, split="train", transform=None)


def test_car_damage_dataset_missing_columns(incomplete_data):
    """
    Test initialization failure when required columns are missing in CSV.
    """
    with pytest.raises(ValueError, match="Missing required columns"):
        CarDamageDataset(data_dir=incomplete_data, split="train", transform=None)


def test_car_damage_dataset_missing_images(missing_images_data):
    """
    Test initialization failure when some images are missing.
    """
    with pytest.raises(FileNotFoundError, match="Missing 1 image files"):
        CarDamageDataset(data_dir=missing_images_data, split="train", transform=None)


def test_car_damage_dataset_len(dummy_data):
    """
    Test the __len__ method of CarDamageDataset.
    """
    dataset = CarDamageDataset(data_dir=dummy_data, split="val", transform=None)
    assert len(dataset) == 5


def test_car_damage_dataset_getitem(dummy_data):
    """
    Test the __getitem__ method of CarDamageDataset.
    """
    transform = None  # No transformations for simplicity
    dataset = CarDamageDataset(data_dir=dummy_data, split="train", transform=transform)
    image, label = dataset[0]
    print(type(image))
    assert isinstance(image, np.ndarray)  # Assuming transform converts to tensor
    assert isinstance(label, np.int64)
    assert label in range(0, 6)  # Labels are 0-based index


def test_car_damage_data_module_setup(dummy_data):
    """
    Test the setup method of CarDamageDataModule.
    """
    datamodule = CarDamageDataModule(data_dir=dummy_data, batch_size=4, num_workers=2, image_size=(256, 256))
    datamodule.setup(stage="fit")
    assert len(datamodule.train_dataset) == 10
    assert len(datamodule.val_dataset) == 5


def test_car_damage_data_module_dataloaders(dummy_data):
    """
    Test the dataloaders from CarDamageDataModule.
    """
    datamodule = CarDamageDataModule(data_dir=dummy_data, batch_size=4, num_workers=2, image_size=(256, 256))
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Check batch size
    batch = next(iter(train_loader))
    images, labels = batch
    assert images.shape[0] == 4  # batch_size
    assert labels.shape[0] == 4


def test_car_damage_data_module_test_loader(dummy_data):
    """
    Test the test_dataloader method of CarDamageDataModule.
    """
    datamodule = CarDamageDataModule(data_dir=dummy_data, batch_size=2, num_workers=1, image_size=(256, 256))
    datamodule.setup(stage="test")

    test_loader = datamodule.test_dataloader()

    assert isinstance(test_loader, DataLoader)

    # Check batch size
    batch = next(iter(test_loader))
    images, labels = batch
    assert images.shape[0] == 2  # batch_size
    assert labels.shape[0] == 2


def test_car_damage_data_module_persistent_workers(dummy_data):
    """
    Test that persistent_workers parameter is set correctly in DataLoaders.
    """
    datamodule = CarDamageDataModule(data_dir=dummy_data, batch_size=4, num_workers=2, image_size=(256, 256))
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    assert train_loader.persistent_workers

    val_loader = datamodule.val_dataloader()
    assert val_loader.persistent_workers

    test_loader = datamodule.test_dataloader()
    assert test_loader.persistent_workers


def test_car_damage_dataset_with_transforms(dummy_data):
    """
    Test that transforms are applied correctly to the dataset.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose(
        [
            A.Resize(128, 128),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    dataset = CarDamageDataset(data_dir=dummy_data, split="train", transform=transform)
    image, label = dataset[0]

    assert image.shape == (3, 128, 128)  # Channels first
    assert isinstance(image, torch.Tensor)
    assert label in range(0, 6)


def test_car_damage_data_module_full_flow(dummy_data):
    """
    Test the full data flow: data loading, transformations, and batching.
    """
    datamodule = CarDamageDataModule(data_dir=dummy_data, batch_size=2, num_workers=1, image_size=(224, 224))
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        images, labels = batch
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] == 2
        break  # Test only the first batch


def test_car_damage_dataset_unique_labels(dummy_data):
    """
    Test that the dataset correctly identifies unique labels.
    """
    dataset = CarDamageDataset(data_dir=dummy_data, split="train", transform=None)
    unique_labels = dataset.df["label"].unique()
    assert len(unique_labels) <= 6  # Assuming num_classes=6
