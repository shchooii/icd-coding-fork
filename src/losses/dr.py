import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal


class DRLoss(nn.Module):
    """Distributionally Robust loss for long‑tailed multi‑label problems.

    Parameters
    ----------
    gamma1 : float, default 5.0
        Scaling for the C‑LSEP pairwise term (controls *hard‑sample* focus).
    gamma2 : float, default 10.0
        Scaling for the NGC term (controls probability threshold for negatives).
    classwise : bool, default True
        If *True* uses class‑wise (C‑LSEP) formulation; otherwise sample‑wise.
    reduction : {"mean", "sum", "none"}, default "mean"
        How to reduce loss across the batch (and classes if class‑wise).
    clamp_min, clamp_max : float, default -30, 30
        Optional clamping of raw logits for numeric stability.
    eps : float, default 1e-6
        Stability epsilon inside ``log1p``.
    """

    def __init__(
        self,
        *,
        gamma1: float = 5.0,
        gamma2: float = 10.0,
        classwise: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
        clamp_min: float = -30.0,
        clamp_max: float = 30.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.classwise = classwise
        self.reduction = reduction
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eps = eps

    # --------------------------------------------------------------------- #
    # Forward helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _masked_logits(logits: Tensor, mask: Tensor, fill_val: float = -1e9) -> Tensor:
        """Return *logits* where entries with *mask==False* are filled with *fill_val*."""
        return logits.masked_fill(~mask, fill_val)

    def _dr_loss_classwise(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute DR loss with **class‑wise** formulation (recommended)."""
        # transpose -> (C, B)
        s: Tensor = logits.t()
        y: Tensor = targets.t().bool()

        pos = self._masked_logits(s, y)          # positive logits shape (C, B_pos)
        neg = self._masked_logits(s, ~y)         # negative logits shape (C, B_neg)

        # Pairwise differences for every (neg, pos) within each class
        diff = (neg.unsqueeze(2) - pos.unsqueeze(1)) * self.gamma1  # (C, N, P)
        pair_term = torch.logsumexp(diff.flatten(start_dim=1), dim=1)  # (C,)

        # Negative‑gradient constraint (NGC)
        ngc_term = torch.logsumexp(self.gamma2 * neg, dim=1)  # (C,)

        loss = torch.log1p(torch.exp(pair_term) + torch.exp(ngc_term) + self.eps)
        return loss  # (C,)

    def _dr_loss_samplewise(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Legacy **sample‑wise** variant (not preferred for heavy imbalance)."""
        s: Tensor = logits
        y: Tensor = targets.bool()

        pos = self._masked_logits(s, y)
        neg = self._masked_logits(s, ~y)

        diff = (neg.unsqueeze(2) - pos.unsqueeze(1)) * self.gamma1  # (B, N, P)
        pair_term = torch.logsumexp(diff.flatten(start_dim=1), dim=1)  # (B,)

        ngc_term = torch.logsumexp(self.gamma2 * neg, dim=1)  # (B,)

        loss = torch.log1p(torch.exp(pair_term) + torch.exp(ngc_term) + self.eps)
        return loss  # (B,)

    # ------------------------------------------------------------------ #
    # Public forward
    # ------------------------------------------------------------------ #

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute DR loss.

        Parameters
        ----------
        logits  : (B, C) raw (pre‑sigmoid) outputs.
        targets : (B, C) binary labels in {0,1}.
        """
        if logits.shape != targets.shape:
            raise ValueError("logits and targets must share shape (B, C)")

        # Clamp for numeric stability (optional)
        logits = logits.clamp(self.clamp_min, self.clamp_max)

        if self.classwise:
            loss = self._dr_loss_classwise(logits, targets)
        else:
            loss = self._dr_loss_samplewise(logits, targets)

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss