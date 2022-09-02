import pytorch_lightning as pl
import torch
import math
import pdb

from torch.nn import Linear

class DST(pl.LightningModule):
    def __init__(
        self,
        n_codes,
        n_vitals,
        d_model,
        n_blocks,
        n_heads,
        dropout,
        pad,
        lr=0.001
    ):
        super().__init__()
        self.save_hyperparameters()

        # embed codes
        # todo: embedding dimension vs. "total" feature space
        # TODO: how to handle padding?
        self.embed_codes = Linear(n_codes, d_model)
        self.embed_vitals = Linear(n_vitals, d_model)
        self.pos_encode = PositionalEncoding(d_model)
        # self.pos_encode = PositionalEncoding(d_model=d_model)
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
            torch.nn.Softmax(),
        )


    def forward(self, batch):
        # static features
        x = self.embed_codes(batch["codes"]).unsqueeze(1)
        # time-varying features
        x = x + self.embed_vitals(batch["vitals"])
        pad_mask = (batch["vitals"][:, :, 0] == self.pad)
        # (max) sequence length
        s = x.shape[1]
        mask = (1 - torch.tril(torch.ones(s, s))).bool().cuda()
        x = self.pos_encode(x)
        x = self.transformer(x, mask, pad_mask)
        # concatenate treatment?
        t = torch.reshape(batch["treatment"], (-1, 1, 1))
        t = t.repeat(1, s, 1)
        # concatenate treatment as a new feature
        x = torch.cat((x, t), 2)
        q_hat = self.to_hazard_c(x).squeeze(2)
        s_hat = q_hat.cumprod(1).clamp(min=1e-8)
        return s_hat



    def training_step(self, batch, batch_idx):
        s_hat =  self(batch)
        loss = self.ordinal_survival_loss(s_hat, batch["survival"])
        self.log("train_loss", loss)
        return loss



    def validation_step(self, batch, batch_idx):
        s_hat =  self(batch)
        loss = self.ordinal_survival_loss(s_hat, batch["survival"])
        self.log("val_loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def ordinal_survival_loss(s_hat, y):
        nlog_survival = -torch.log(s_hat)
        nlog_failure = -torch.log(1 - s_hat)
        return nlog_survival[y == 1].mean() + nlog_failure[y == 0].mean()


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
