import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ml_ops_project.data import CarDamageDataModule
from ml_ops_project.model import CarDamageModel


@hydra.main(
    version_base="1.1",
    config_path=str(Path(__file__).parents[2] / "configs"),  # up 2 parents from train.py to get to your_repo/
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Initialize Wandb Logger with environment variables, falling back to config values
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT", cfg.logger.wandb.project),
        entity=os.getenv("WANDB_ENTITY", cfg.logger.wandb.entity),
        log_model=cfg.logger.wandb.log_model,  # Keep this from config as it's not sensitive
    )
    # Note: WANDB_API_KEY is automatically picked up by wandb from environment

    # Rest of your code remains the same
    data_module = CarDamageDataModule(
        data_dir=Path(cfg.data.data_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=tuple(cfg.data.image_size),
    )
    data_module.setup()

    # Initialize Model
    model = CarDamageModel(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        scheduler_step_size=cfg.model.scheduler.step_size,
        scheduler_gamma=cfg.model.scheduler.gamma,
    )

    # Initialize Model Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.trainer.callbacks.checkpoint.monitor,
        dirpath=cfg.trainer.callbacks.checkpoint.dirpath,
        filename=cfg.trainer.callbacks.checkpoint.filename,
        save_top_k=cfg.trainer.callbacks.checkpoint.save_top_k,
        mode=cfg.trainer.callbacks.checkpoint.mode,
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Start Training
    trainer.fit(model, datamodule=data_module)

    # Test the model on the test set
    trainer.test(model, datamodule=data_module)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    main()
