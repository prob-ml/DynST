import pytorch_lightning as pl
import torch

class BaseModel(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
        # return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)