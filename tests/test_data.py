from torch.utils.data import Dataset
from pathlib import Path

from ml_ops_project.data import CarDamageDataset


# def test_my_dataset():
#     """Test the MyDataset class."""
#     dataset = CarDamageDataset(Path("data/processed"),
#                 split='train',
#                 transform=None)
#     assert isinstance(dataset, Dataset)
