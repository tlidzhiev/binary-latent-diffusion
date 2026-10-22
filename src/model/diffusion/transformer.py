import math
from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torchvision.ops import StochasticDepth


def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2)
    omega = 1.0 / 10000**omega
    out = torch.outer(pos.reshape(-1).to(torch.float64), omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    coords = torch.arange(grid_size, dtype=torch.float64)
    rows, cols = torch.meshgrid(coords, coords, indexing='ij')
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, cols)
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, rows)
    return torch.cat([emb_w, emb_h], dim=1).float()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, num_timesteps: int, dim: int, rescale_steps: int = 4000) -> None:
        super().__init__()
        self.dim = dim
        self.num_timesteps = float(num_timesteps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t / self.num_timesteps * self.rescale_steps
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device) * -(math.log(10000) / (half_dim - 1))
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = rearrange(self.qkv(x), 'b l (qkv h d) -> qkv b h l d', qkv=3, h=self.num_heads)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'b h l d -> b l (h d)')
        return self.proj(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(context_dim, 2 * embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.null_context = nn.Parameter(torch.randn(1, 1, context_dim) * 0.01)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if context is None:
            context = self.null_context.expand(x.shape[0], -1, -1)
        q = rearrange(self.q(x), 'b l (h d) -> b h l d', h=self.num_heads)
        k, v = rearrange(self.kv(context), 'b l (kv h d) -> kv b h l d', kv=2, h=self.num_heads)
        x = nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'b h l d -> b l (h d)')
        return self.proj(x)


class Block(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_heads, dropout)
        self.gamma1 = nn.Parameter(torch.ones(embed_dim))

        self.cross_attn = None
        if context_dim is not None:
            self.norm_cross = nn.LayerNorm(embed_dim)
            self.cross_attn = CrossAttention(embed_dim, num_heads, context_dim, dropout)
            self.gamma_cross = nn.Parameter(torch.ones(embed_dim))

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.gamma2 = nn.Parameter(torch.ones(embed_dim))
        self.drop_path = StochasticDepth(drop_path, mode='row')

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        if self.cross_attn is not None:
            x = x + self.drop_path(self.gamma_cross * self.cross_attn(self.norm_cross(x), context))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class DenoisingTransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 64,
        latent_size: int = 16,
        num_timesteps: int = 64,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 24,
        embed_dropout: float = 0.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        condition: Literal['none', 'class', 'text', 'text_cross'] = 'none',
        num_classes: int | None = None,
        text_embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.condition = condition
        self.codebook_size = codebook_size
        self.latent_size = latent_size
        self.num_timesteps = num_timesteps

        self.token_proj = nn.Linear(codebook_size, embed_dim, bias=False)
        nn.init.normal_(self.token_proj.weight, mean=0.0, std=0.02)
        self.pos_emb = nn.Parameter(get_2d_sincos_pos_embed(embed_dim, latent_size).unsqueeze(0))
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(num_timesteps, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        match condition:
            case 'class':
                assert num_classes is not None, 'num_classes is required for class conditioning'
                self.cond_emb = nn.Embedding(num_classes, embed_dim)
            case 'text':
                self.cond_emb = nn.Linear(text_embed_dim, embed_dim)
            case _:
                self.cond_emb = None
        self.drop = nn.Dropout(embed_dropout)

        context_dim = text_embed_dim if condition == 'text_cross' else None
        drop_path_rates = torch.linspace(0, drop_path, num_layers).tolist()
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, dropout, rate, context_dim) for rate in drop_path_rates]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, codebook_size),
        )

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = xt.shape[1]
        x = self.token_proj(xt * 2.0 - 1.0) + self.pos_emb[:, :num_tokens]
        tokens = [x, self.time_emb(t).unsqueeze(1)]
        if self.cond_emb is not None and cond is not None:
            tokens.append(self.cond_emb(cond).unsqueeze(1))
        x = self.drop(torch.cat(tokens, dim=1))

        context = cond if self.condition == 'text_cross' else None
        for block in self.blocks:
            x = block(x, context)
        return self.head(x[:, :num_tokens])
