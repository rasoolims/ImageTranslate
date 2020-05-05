import torch.nn as nn


class SmoothedNLLLoss(nn.NLLLoss):
    def __init__(self, weight=None, ignore_index=-100, reduce: bool = False, epsilon=0.1):
        super().__init__(weight=weight, ignore_index=ignore_index)
        self.epsilon = epsilon
        self.reduce = False

    def forward(self, input, target):
        if target.dim() == input.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -input.gather(dim=-1, index=target)
        smooth_loss = -input.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.epsilon / input.size(-1)
        loss = (1. - self.epsilon) * nll_loss + eps_i * smooth_loss
        return loss
