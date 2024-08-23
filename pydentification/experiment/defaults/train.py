import lightning.pytorch as pl


def train_fn(
    model: pl.LightningModule, trainer: pl.Trainer, dm: pl.LightningDataModule, checkpoint_path: str | None = None
) -> tuple[pl.LightningModule, pl.Trainer]:
    """
    Runs training using pl.Trainer and pl.LightningModule with given LightningDataModule, returns both model and trainer
    Can be restarted if checkpoint_path is provided.
    """
    trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path)
    return model, trainer
