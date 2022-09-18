import torch
from torchmetrics import Metric


class MeanAbsoluteError(Metric):
    full_state_update = True
    higher_is_better = False
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        self.add_state("error", default=torch.tensor(0.))
        self.add_state("total", default=torch.tensor(0.))

    def update(self, s_hat, y):
        observed = (y == 0).any(1).int()
        t_hat = torch.where(y == self.pad, 0, s_hat).sum(1)
        t = torch.where(y == self.pad, 0, y).sum(1)
        zeros = torch.zeros(t.shape).cuda()
        observed_error = torch.abs(t_hat - t) * observed
        censored_error = torch.maximum(zeros, t - t_hat) * (1 - observed)
        self.error +=  observed_error.sum() + censored_error.sum()
        self.total += t.numel()

    def compute(self):
        return self.error / self.total

class ConcordanceIndex(Metric):
    higher_is_better = True
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        self.add_state("observed", default=[])
        self.add_state("true", default=[])
        self.add_state("predicted", default=[])

    def update(self, s_hat, y):
        self.observed.append((y == 0).any(1).int())
        self.true.append(torch.where(y == self.pad, 0, y).sum(1))
        self.predicted.append(torch.where(y == self.pad, 0, s_hat).sum(1))

    def compute(self):
        assert len(self.true) > 1
        # get pairs of elements
        t = torch.combinations(torch.cat(self.true))
        t_hat = torch.combinations(torch.cat(self.predicted))
        d = torch.combinations(torch.cat(self.observed))
        num = (
            (t[:, 0] < t[:, 1]) * (t_hat[:, 0] < t_hat[:, 1]) *\
                d[:, 0]
        ).sum()
        denom = (
            (t[:, 0] < t[:, 1]) * d[:, 0]
        ).sum()
        return num / denom
