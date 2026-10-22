from typing import Self

import torch
from torch import nn

from src.model.bae import BinaryAutoEncoder


class BinaryEncode(nn.Module):
    def __init__(self, autoencoder: BinaryAutoEncoder, deterministic: bool = False) -> None:
        super().__init__()
        self.autoencoder = autoencoder.eval().requires_grad_(False)
        self.deterministic = deterministic

    def train(self, mode: bool = True) -> Self:
        # the autoencoder always stays in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(
            img,
            deterministic=self.deterministic,
            code_only=True,
        )
