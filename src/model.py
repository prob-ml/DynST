import pytorch_lightning as pl
import torch
import math
import pdb

from torch.nn import Linear
from .metric import MeanAbsoluteError, ConcordanceIndex

class DST(pl.LightningModule):
    def __init__(
        self,
        n_codes,
        n_vitals,
        n_demog,
        d_model,
        n_blocks,
        n_heads,
        dropout,
        pad,
        dynamic,
        lr=0.001,
        alpha=0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        # TODO: incorporate demographic info
        self.embed_codes = Linear(n_codes, d_model)
        self.embed_static = Linear(d_model + n_demog, d_model)
        self.embed_vitals = Linear(n_vitals, d_model)
        self.pos_encode = PositionalEncoding(d_model)
        self.pad = pad
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model*4
        )
        norm = torch.nn.LayerNorm(d_model)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, n_blocks, norm)
        self.to_hazard_c = torch.nn.Sequential(
            Linear(d_model+1, d_model//2),
            torch.nn.ReLU(),
            Linear(d_model//2, 1),
            torch.nn.Sigmoid(),
        )
        self.train_mae = MeanAbsoluteError(pad=pad)
        self.val_mae = MeanAbsoluteError(pad=pad)
        self.val_ci = ConcordanceIndex(pad=pad)
        # how much to weigh MAE loss
        self.alpha = alpha
        self.dynamic = dynamic


    def forward(self, batch):
        # static features
        x = self.embed_codes(batch["codes"]).unsqueeze(1)
        x = self.embed_static(
            torch.cat([x, batch["demog"].unsqueeze(1)], 2)
        )
        s = batch["vitals"].shape[1]
        # time-varying features
        if self.dynamic:
            pad_mask = (batch["vitals"][:, :, 0] == self.pad)
            x = x + self.embed_vitals(batch["vitals"])
            # autoregressive mask
            mask = (1 - torch.tril(torch.ones(s, s))).bool().cuda()

        else:
            mask = None
            x = x.repeat(1, s, 1)
            pad_mask = (batch["vitals"][:, :, 0] == self.pad)
        x = self.pos_encode(x)
        x = self.transformer(x, mask, pad_mask)
        # concatenate treatment?
        t = torch.reshape(batch["treatment"], (-1, 1, 1))
        t = t.repeat(1, s, 1)
        # concatenate treatment as a new feature
        x = torch.cat((x, t), 2).float()
        # complement of hazard
        q_hat = self.to_hazard_c(x).squeeze(2)
        s_hat = q_hat.cumprod(1).clamp(min=1e-8)
        return s_hat



    def training_step(self, batch, batch_idx):
        s_hat =  self(batch)
        loss = self.combined_loss(s_hat, batch["survival"])
        # loss = self.loss_fn(s_hat, batch["survival"])
        self.log("train_loss", loss)
        self.train_mae(s_hat, batch["survival"])
        self.log("train_mae", self.train_mae, on_step=True, on_epoch=False)
        return loss



    def validation_step(self, batch, batch_idx):
        s_hat =  self(batch)
        loss = self.combined_loss(s_hat, batch["survival"])
        # loss = self.loss_fn(s_hat, batch["survival"])
        self.val_mae.update(s_hat, batch["survival"])
        self.val_ci.update(s_hat, batch["survival"])
        # TODO: separately log ordinal loss
        self.log("val_loss", loss)
        self.log("val_mae", self.val_mae, on_step=True, on_epoch=True)
        self.log("val_ci", self.val_ci, on_step=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        # returns estimated survival times
        s_hat = self(batch)
        mask = (batch["survival"] != self.pad)
        return (s_hat * mask).sum(1)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def ordinal_survival_loss(self, s_hat, y):
        # modified cross entropy loss
        nlog_survival = -torch.log(s_hat)
        nlog_failure = -torch.log(1 - s_hat)
        loss = 0
        loss += nlog_survival * torch.where(y==self.pad, 0, y)
        loss += nlog_failure * torch.where(y==self.pad, 0, (1-y))
        return loss.sum() / (y != self.pad).sum()
    
    def mae_loss(self, s_hat, y):
        observed = (y == 0).any(1).int()
        t_hat = torch.where(y == self.pad, 0, s_hat).sum(1)
        t = torch.where(y == self.pad, 0, y).sum(1)
        zeros = torch.zeros(t.shape).cuda()
        observed_error = torch.abs(t_hat - t) * observed
        censored_error = torch.maximum(zeros, t - t_hat) * (1 - observed)
        return (observed_error.sum() + censored_error.sum()) / t.numel()

    def combined_loss(self, s_hat, y):
        a = self.alpha
        ordinal_loss = self.ordinal_survival_loss(s_hat, y)
        mae_loss = self.mae_loss(s_hat, y)
        return (1 - a) * ordinal_loss + a * mae_loss

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)
