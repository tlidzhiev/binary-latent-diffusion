from pathlib import Path
from typing import Literal, overload

import torch
from einops import rearrange
from torch import nn

from ..base import BaseModel
from .utils import convert_state_dict


class BinaryQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim, codebook_size, 1),
            nn.Sigmoid(),
        )
        self.embed = nn.Embedding(codebook_size, embed_dim)

    def _quantize(self, probs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return torch.gt(probs, 0.5).to(probs.dtype) if deterministic else torch.bernoulli(probs)

    def embed_code(self, code: torch.Tensor) -> torch.Tensor:
        return torch.einsum('b n h w, n d -> b d h w', code, self.embed.weight)

    def get_code(self, h: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self._quantize(self.proj(h), deterministic=deterministic)

    def forward(
        self, h: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = self.proj(h)
        z_b = self._quantize(probs, deterministic=deterministic)
        z_flow = z_b.detach() + probs - probs.detach()
        z_q = self.embed_code(z_flow)
        return z_q, z_b, probs


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.qkv = nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = rearrange(self.qkv(h_), 'b (qkv c) h w -> qkv b 1 (h w) c', qkv=3)
        h_ = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h_ = rearrange(h_, 'b 1 (h w) c -> b c h w', h=x.shape[2], w=x.shape[3])
        return x + self.proj_out(h_)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: list[int],
    ) -> None:
        super().__init__()
        num_resolutions = len(channel_mults)
        curr_res = resolution
        in_channel_mults = [1, *channel_mults]

        blocks: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        ]
        block_in_ch = base_channels
        for i in range(num_resolutions):
            block_in_ch = base_channels * in_channel_mults[i]
            block_out_ch = base_channels * channel_mults[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(SelfAttentionBlock(block_in_ch))
            if i != num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        blocks.extend(
            [
                ResBlock(block_in_ch, block_in_ch),
                SelfAttentionBlock(block_in_ch),
                ResBlock(block_in_ch, block_in_ch),
                nn.GroupNorm(num_groups=32, num_channels=block_in_ch, eps=1e-6, affine=True),
                nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
        channel_mults: list[int],
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: list[int],
    ) -> None:
        super().__init__()
        num_resolutions = len(channel_mults)
        block_in_ch = base_channels * channel_mults[-1]
        curr_res = resolution // 2 ** (num_resolutions - 1)

        blocks: list[nn.Module] = [
            nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True),
            nn.Conv2d(in_channels, block_in_ch, kernel_size=3, stride=1, padding=1),
            ResBlock(block_in_ch, block_in_ch),
            SelfAttentionBlock(block_in_ch),
            ResBlock(block_in_ch, block_in_ch),
        ]
        for i in reversed(range(num_resolutions)):
            block_out_ch = base_channels * channel_mults[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(SelfAttentionBlock(block_in_ch))
            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        blocks.extend(
            [
                nn.GroupNorm(num_groups=32, num_channels=block_in_ch, eps=1e-6, affine=True),
                nn.Conv2d(block_in_ch, out_channels, kernel_size=3, stride=1, padding=1),
            ]
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class BinaryAutoEncoder(BaseModel):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        out_channels: int = 3,
        channel_mults: list[int] | None = None,
        num_res_blocks: int = 2,
        resolution: int = 256,
        attn_resolutions: list[int] | None = None,
        codebook_size: int = 64,
        embed_dim: int = 256,
        decoder_mult: int = 1,
    ) -> None:
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 1, 2, 2, 4]
        if attn_resolutions is None:
            attn_resolutions = [16]
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=embed_dim,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
        )
        self.quantizer = BinaryQuantizer(codebook_size, embed_dim, embed_dim)
        self.decoder = Decoder(
            in_channels=embed_dim,
            base_channels=base_channels,
            out_channels=out_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks * decoder_mult,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
        )

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, **kwargs) -> 'BinaryAutoEncoder':
        """
        Build the autoencoder and load weights of the original BinaryGAN checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the checkpoint, relative to the project root or absolute.
        **kwargs
            Architecture arguments, e.g. `codebook_size=128, decoder_mult=3` for C128.

        Returns
        -------
        BinaryAutoEncoder
            Model with the loaded weights.
        """
        model = cls(**kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(convert_state_dict(state_dict), strict=True)
        return model

    @property
    def last_layer(self) -> torch.Tensor:
        last = self.decoder.blocks[-1]
        assert isinstance(last, nn.Conv2d)
        return last.weight

    @overload
    def encode(
        self, img: torch.Tensor, deterministic: bool = False, *, code_only: Literal[True]
    ) -> torch.Tensor: ...

    @overload
    def encode(
        self, img: torch.Tensor, deterministic: bool = False, *, code_only: Literal[False] = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def encode(
        self, img: torch.Tensor, deterministic: bool = False, *, code_only: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(img)
        if code_only:
            return self.quantizer.get_code(h, deterministic=deterministic)
        return self.quantizer(h, deterministic=deterministic)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantizer.embed_code(code)).clip(0, 1)

    def forward(
        self,
        img: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        z_q, code, probs = self.encode(img, deterministic=deterministic)
        recon = self.decoder(z_q).clip(0, 1)
        return {'recon': recon, 'code': code, 'probs': probs}
