import torch
from einops import rearrange

from ..base import BaseModel
from .scheduler import NoiseScheduler
from .transformer import DenoisingTransformer


class BinaryDiffusion(BaseModel):
    def __init__(
        self,
        denoiser: DenoisingTransformer,
        num_timesteps: int = 64,
        schedule: str = 'linear',
        p_flip: bool = True,
        cond_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert denoiser.num_timesteps == num_timesteps, (
            f'Denoiser has {denoiser.num_timesteps} timesteps, expected {num_timesteps}.'
        )
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.p_flip = p_flip
        self.cond_dropout = cond_dropout
        self.scheduler = NoiseScheduler(num_timesteps, schedule)

    def forward(
        self,
        img: torch.Tensor,
        label: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        x0 = rearrange(img, 'b c h w -> b (h w) c').float()
        t = torch.randint(1, self.num_timesteps + 1, (x0.shape[0],), device=x0.device)
        xt = torch.bernoulli(self.scheduler(x0, t))

        cond = label if self.denoiser.condition != 'none' else None
        if self.training and self.cond_dropout > 0 and torch.rand(()).item() < self.cond_dropout:
            cond = None

        logits = self.denoiser(xt, t - 1, cond)
        return {'logits': logits, 'x0': x0, 'xt': xt, 't': t}

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        cond: torch.Tensor | None = None,
        temperature: float = 1.0,
        num_steps: int | None = None,
        guidance: float | None = None,
    ) -> torch.Tensor:
        num_steps = num_steps or self.num_timesteps
        assert 1 <= num_steps <= self.num_timesteps, (
            f'num_steps must be in [1, {self.num_timesteps}]'
        )
        device = next(self.denoiser.parameters()).device
        cond = cond if self.denoiser.condition != 'none' else None

        size, codebook_size = self.denoiser.latent_size, self.denoiser.codebook_size
        xt = torch.bernoulli(torch.full((num_samples, size**2, codebook_size), 0.5, device=device))

        # evenly spaced subset of the steps (integer arithmetic, no float rounding), descending
        idx = torch.arange(num_steps) * (self.num_timesteps - 1) // max(num_steps - 1, 1)
        steps = (idx + 1).flip(0).tolist()

        for i, step in enumerate(steps):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            logits = self.denoiser(xt, t - 1, cond)
            if guidance is not None and cond is not None:
                logits = (1 + guidance) * logits - guidance * self.denoiser(xt, t - 1, None)
            probs = torch.sigmoid(logits / temperature)
            # probability of x_0 = 1
            x0_probs = xt * (1 - probs) + (1 - xt) * probs if self.p_flip else probs

            if step == 1:
                xt = (x0_probs > 0.5).float()
            else:
                s = torch.full_like(t, steps[i + 1])
                xt = torch.bernoulli(self.scheduler.posterior(x0_probs, xt, t, s))

        return rearrange(xt, 'b (h w) c -> b c h w', h=size, w=size)
