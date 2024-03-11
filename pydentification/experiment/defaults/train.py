import lightning.pytorch as pl


def train_fn(
    model: pl.LightningModule, trainer: pl.Trainer, dm: pl.LightningDataModule
) -> tuple[pl.LightningModule, pl.Trainer]:
    """
    Runs training using pl.Trainer and pl.LightningModule
    with given LightningDataModule, returns both model and trainer
    """
    trainer.fit(model, datamodule=dm)
    return model, trainer
