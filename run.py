import hydra
from hydra.utils import get_original_cwd, instantiate
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.preprocess import Mimic3Pipeline
from src.dataset import Mimic3Dataset, padded_collate
  
@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    owd = get_original_cwd()
    if cfg.preprocess.do:
        pipeline = Mimic3Pipeline(owd, seed=cfg.preprocess.seed)
        pipeline.run()
        return
    dataset = Mimic3Dataset(owd, seed=cfg.preprocess.seed)
    train_frac = cfg.train.train_frac
    val_frac = cfg.train.val_frac
    train_size = int(train_frac * len(dataset))
    if train_frac + val_frac == 1.0:
        val_size = len(dataset) - train_size
        test_size = 0
    else:
        val_size = int(val_frac * len(dataset))
        test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset,
        (train_size, val_size, test_size),
        torch.Generator().manual_seed(cfg.train.seed)
    )
    def collate(x):
        return padded_collate(x, pad_index=cfg.model.pad, causal=cfg.causal)

    train_loader = DataLoader(
        train_set,
        collate_fn=collate,
        batch_size=cfg.train.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set, collate_fn=collate, batch_size=cfg.train.batch_size
    )
    if test_size:
        test_loader = DataLoader(
            test_set, collate_fn=collate, batch_size=cfg.train.batch_size
        )
    model = instantiate(
        cfg.model, n_codes=dataset.n_codes, n_demog = dataset.n_demog, 
        n_vitals=dataset.n_vitals, causal=cfg.causal,
    )
    callbacks = [ModelCheckpoint(monitor="val_mae_epoch", mode="min")]
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
    if test_size:
        trainer.test(dataloaders=test_loader)


    # if cfg.causal:
    #     dataset_treated = Mimic3Dataset(owd, seed=cfg.preprocess.seed, intervention=True)
    #     dataset_control = Mimic3Dataset(owd, seed=cfg.preprocess.seed, intervention=False)
    #     treated_loader = DataLoader(dataset_treated, collate_fn=collate, batch_size=96)
    #     control_loader = DataLoader(dataset_control, collate_fn=collate, batch_size=96)
    #     predict_t = torch.cat(trainer.predict(dataloaders=treated_loader))
    #     torch.save(predict_t, "t_hat_treated.pt")
    #     predict_c = torch.cat(trainer.predict(dataloaders=control_loader))
    #     torch.save(predict_c, "t_hat_control.pt")

        



if __name__ == "__main__":
    main()