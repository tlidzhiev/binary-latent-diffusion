from typing import Literal

import torch
from torch import nn

from src.model.diffusion import NoiseScheduler

from .base import BaseLoss


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float | Literal['balanced'] | None = 'balanced',
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = loss * (1 - p_t).clamp(min=1e-6, max=1 - 1e-6) ** gamma
    if alpha == 'balanced':
        pos_frac = targets.sum((-1, -2), keepdim=True) / targets[0].numel()
        alpha_t = (1 - pos_frac) * targets + pos_frac * (1 - targets)
        loss = alpha_t * loss
    elif alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


class BinaryDiffusionLoss(BaseLoss):
    def __init__(
        self,
        num_timesteps: int = 64,
        schedule: str = 'linear',
        p_flip: bool = True,
        focal_gamma: float | None = 0.0,
        focal_alpha: float | Literal['balanced'] | None = 'balanced',
        weighting: Literal['mean', 'weighted'] = 'mean',
        aux_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.p_flip = p_flip
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.weighting = weighting
        self.aux_weight = aux_weight
        self.scheduler = NoiseScheduler(num_timesteps, schedule)

        self.loss_names = ['loss', 'x0_loss']
        if aux_weight > 0:
            self.loss_names.append('aux_loss')

    def forward(
        self,
        logits: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        x0 = x0.float()
        target = torch.logical_xor(x0, xt).float() if self.p_flip else x0
        if self.focal_gamma is None:
            x0_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction='none'
            )
        else:
            x0_loss = focal_loss(logits, target, gamma=self.focal_gamma, alpha=self.focal_alpha)

        if self.weighting == 'weighted':
            weight = (1 - (t - 1) / self.num_timesteps).view(-1, 1, 1)
        else:
            weight = torch.ones_like(t, dtype=x0_loss.dtype).view(-1, 1, 1)

        losses = {'x0_loss': x0_loss.mean()}
        loss = (weight * x0_loss).mean()

        if self.aux_weight > 0:
            x0_logits = logits * (1 - 2 * xt) if self.p_flip else logits
            x0_probs = torch.sigmoid(x0_logits)
            is_last = (t == 1).view(-1, 1, 1)
            pred = torch.where(is_last, x0_probs, self.scheduler.posterior(x0_probs, xt, t))
            gt = self.scheduler.posterior(x0, xt, t)
            aux_loss = nn.functional.binary_cross_entropy(
                pred.clamp(min=1e-6, max=1 - 1e-6), gt.clamp(min=0.0, max=1.0), reduction='none'
            )
            losses['aux_loss'] = (weight * aux_loss).mean()
            loss = loss + self.aux_weight * losses['aux_loss']

        losses['loss'] = loss
        return losses
