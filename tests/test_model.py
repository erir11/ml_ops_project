import pytest
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from ml_ops_project.model import CarDamageModel  # adjust import path as needed

@pytest.fixture
def mock_batch():
    """Create a mock batch of images and labels."""
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    num_classes = 6
    
    # Create random images and random labels
    images = torch.randn(batch_size, channels, height, width)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return images, labels

@pytest.fixture
def model():
    """Create a model instance."""
    return CarDamageModel(
        model_name="resnet50",
        num_classes=6,
        learning_rate=1e-3
    )

def test_model_init(model):
    """Test model initialization."""
    assert isinstance(model, LightningModule)
    assert model.hparams.num_classes == 6
    assert model.hparams.learning_rate == 1e-3
    assert model.hparams.model_name == "resnet50"

def test_forward_pass(model, mock_batch):
    """Test the forward pass of the model."""
    images, _ = mock_batch
    outputs = model(images)
    
    assert outputs.shape == (2, 6)  # (batch_size, num_classes)
    assert not torch.isnan(outputs).any(), "Output contains NaN values"

def test_training_step(model, mock_batch):
    """Test the training step."""
    loss = model.training_step(mock_batch, batch_idx=0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss).any(), "Loss contains NaN values"

def test_validation_step(model, mock_batch):
    """Test the validation step."""
    model.validation_step(mock_batch, batch_idx=0)
    # No explicit assert needed as validation_step doesn't return anything
    # If no exception is raised, the test passes

def test_configure_optimizers(model):
    """Test optimizer and scheduler configuration."""
    optimizers, schedulers = model.configure_optimizers()
    
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    assert isinstance(optimizers[0], torch.optim.Adam)
    assert isinstance(schedulers[0], torch.optim.lr_scheduler.StepLR)

def test_model_output_shape(model):
    """Test if model produces correct output shape for various batch sizes."""
    batch_sizes = [1, 4, 8]
    for batch_size in batch_sizes:
        images = torch.randn(batch_size, 3, 224, 224)
        outputs = model(images)
        assert outputs.shape == (batch_size, 6), f"Failed for batch size {batch_size}"