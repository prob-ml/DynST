import hydra
from hydra.utils import get_original_cwd, instantiate
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.preprocess import Mimic3Pipeline
from src.dataset import Mimic3Dataset, padded_collate
import pdb
  
@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    owd = get_original_cwd()
    if cfg.preprocess.do:
        pipeline = Mimic3Pipeline(owd)
        pipeline.run()
        return
    dataset = Mimic3Dataset(owd)
    train_size = int(cfg.train.train_frac * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        (train_size, val_size),
        torch.Generator().manual_seed(cfg.seed)
    )
    collate_fn = lambda x: padded_collate(x, cfg.model.pad)
    train_loader = DataLoader(
        train_set,
        collate_fn = collate_fn,
        batch_size = cfg.train.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set, collate_fn = collate_fn, batch_size = cfg.train.batch_size
    )
    model = instantiate(
        cfg.model, n_codes=dataset.n_codes, n_demog = dataset.n_demog, 
        n_vitals=dataset.n_vitals,
    )
    callbacks = [ModelCheckpoint(monitor="val_ci", mode="max")]
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)





    # build data loaders for train & val
    # create model
    # initialize trainer



if __name__ == "__main__":
    main()