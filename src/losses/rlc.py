import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReflectiveLabelCorrectorLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        distribution,
        tau: float = 0.7,
        compute_epoch: int = 2,
        margin: float = 1.0,
        gamma_pos: float = 0.0,
        gamma_neg: float = 3.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()

        dist = torch.tensor(
            distribution,
            dtype=torch.float32,
        )
        if len(dist) != num_classes:
            raise ValueError("distribution length ≠ num_classes")

        self.register_buffer("dist", dist.clone())
        self.register_buffer("weight", self._make_weight(dist, eps))

        self.tau = tau
        self.compute_epoch = compute_epoch
        self.margin = margin
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

        self.total_samples = dist.sum()
        self.epoch = 0

    @staticmethod
    def _make_weight(dist: torch.Tensor, eps: float) -> torch.Tensor:
        p = dist / (dist.sum() + eps)
        p = p / (p.max() + eps)
        return torch.pow(-torch.log(p.clamp_min(eps)) + 1.0, 1.0 / 6)

    @torch.no_grad()
    def _update_weight(self, flipped: torch.Tensor) -> None:
        self.dist.add_(flipped.sum(dim=0))
        self.weight.copy_(self._make_weight(self.dist, self.eps))

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        self.epoch += 1

        logits = logits - self.margin * targets
        preds = torch.sigmoid(logits)
        avg_preds = preds.mean(dim=0, keepdim=True)

        if self.epoch <= self.compute_epoch:
            return logits.sum() * 0.0

        corrected = (preds > self.tau) & (preds > avg_preds)
        new_targets = torch.where(corrected, torch.ones_like(targets), targets)
        flipped = new_targets - targets
        sum_pos = new_targets.sum().clamp_min(1)

        self._update_weight(flipped)

        pos_term = new_targets * torch.log(preds.clamp_min(self.eps))
        neg_term = (1 - new_targets) * torch.log((1 - preds + self.clip).clamp_max(1.0).clamp_min(self.eps))
        loss_mat = (pos_term + neg_term) * self.weight

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            with torch.no_grad():
                pt = preds * new_targets + (1 - preds) * (1 - new_targets)
                gamma = self.gamma_pos * new_targets + self.gamma_neg * (1 - new_targets)
                mod = torch.pow(1 - pt, gamma)
            loss_mat = loss_mat * mod

        loss = -loss_mat.sum() * (logits.size(0) / float(sum_pos))
        return loss
