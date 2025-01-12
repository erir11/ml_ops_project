import shutil
from pathlib import Path
from typing import Dict

import kagglehub
import pandas as pd
import typer
from sklearn.model_selection import train_test_split


def download_dataset(raw_path: Path) -> None:
    """Download the dataset if it doesn't exist."""
    if _is_raw_data_present(raw_path):
        print(f"Dataset already exists in {raw_path}")
        return
    
    raw_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading dataset...")
    kaggle_path = kagglehub.dataset_download(
        "sharko294/image-dataset-for-car-damage-classification"
    )
    
    print(f"Moving dataset to {raw_path}...")
    for file_path in Path(kaggle_path).glob("*"):
        dest_path = raw_path / file_path.name
        shutil.move(str(file_path), str(dest_path))
    
    Path(kaggle_path).rmdir()

def create_splits(
    raw_path: Path,
    processed_path: Path,
    splits: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15},
    force: bool = False
) -> None:
    """Create train/val/test splits from raw data."""
    if _are_splits_present(processed_path) and not force:
        print("Splits already exist. Use --force to recreate.")
        return
        
    # Read the raw data
    df = pd.read_csv(raw_path / "train" / "train.csv")
    
    # Create split DataFrames
    train_df, temp_df = train_test_split(
        df,
        train_size=splits["train"],
        random_state=42,
        stratify=df['label'] if 'label' in df.columns else None
    )
    
    # Split remaining data into val and test
    remaining_ratio = splits["val"] / (splits["val"] + splits["test"])
    val_df, test_df = train_test_split(
        temp_df,
        train_size=remaining_ratio,
        random_state=42,
        stratify=temp_df['label'] if 'label' in temp_df.columns else None
    )
    
    # Save splits
    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    
    for split_name, split_df in split_dfs.items():
        _save_split(
            split_df,
            raw_path / "train" / "images",
            processed_path / split_name,
            split_name
        )

def _is_raw_data_present(path: Path) -> bool:
    """Check if raw dataset exists."""
    return (path / "train" / "train.csv").exists() and (path / "train" / "images").exists()

def _are_splits_present(path: Path) -> bool:
    """Check if splits exist."""
    splits = ["train", "val", "test"]
    return all(
        (path / split / f"{split}.csv").exists() and 
        (path / split / "images").exists() and
        any((path / split / "images").iterdir())
        for split in splits
    )

def _save_split(df: pd.DataFrame, src_images: Path, output_dir: Path, split_name: str) -> None:
    """Save a data split and its images."""
    print(f"Processing {split_name} split...")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Save CSV
    df.to_csv(output_dir / f"{split_name}.csv", index=False)
    
    # Copy images
    for _, row in df.iterrows():
        image_filename = row['image_path'] if 'image_path' in row else row['filename']
        src_path = src_images / image_filename
        dst_path = images_dir / image_filename
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Image not found: {src_path}")

def prepare_dataset(
    raw_path: Path = Path("data/raw"),
    processed_path: Path = Path("data/processed"),
    train_size: float = 0.7,
    val_size: float = 0.15,
    force: bool = False
) -> None:
    """Download and prepare the dataset."""
    download_dataset(raw_path)
    splits = {
        "train": train_size,
        "val": val_size,
        "test": 1 - train_size - val_size
    }
    create_splits(raw_path, processed_path, splits, force)

if __name__ == "__main__":
    typer.run(prepare_dataset)
