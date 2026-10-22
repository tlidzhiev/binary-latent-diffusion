import torch
from torch import nn


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 64,
        schedule: str = 'linear',
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        match schedule:
            case 'linear':
                beta = torch.cat(
                    [torch.ones(1, dtype=torch.float64), 1 - 1 / (num_timesteps - steps[1:] + 1)]
                )
                k_final = torch.cumprod(beta, dim=0)
            case 'cos':
                k_final = 0.5 + 0.5 * torch.cos(torch.pi * steps / num_timesteps)
                beta = torch.cat([torch.ones(1, dtype=torch.float64), k_final[1:] / k_final[:-1]])
            case 'sigmoid':
                k_final = self._sigmoid_schedule(steps / num_timesteps, start=0, end=3, tau=0.8)
                beta = torch.cat([torch.ones(1, dtype=torch.float64), k_final[1:] / k_final[:-1]])
            case _:
                raise ValueError(f"Unknown noise schedule: '{schedule}'.")
        b_final = (1 - k_final) / 2

        self.register_buffer('k_final', k_final.float())
        self.register_buffer('b_final', b_final.float())
        self.register_buffer('beta', beta.float())

    @staticmethod
    def _sigmoid_schedule(
        t: torch.Tensor,
        start: float,
        end: float,
        tau: float,
        clip_min: float = 0.0,
    ) -> torch.Tensor:
        v_start = torch.sigmoid(torch.tensor(start / tau, dtype=t.dtype))
        v_end = torch.sigmoid(torch.tensor(end / tau, dtype=t.dtype))
        output = torch.sigmoid((t * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start)
        return output.clamp(min=clip_min, max=1.0)

    @staticmethod
    def _expand(coef: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return coef.view(-1, *([1] * (x.ndim - 1)))

    def jump(self, x: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # q(x_t | x_s): t - s noising steps, the product of betas is taken explicitly
        # instead of k_t / k_s, which is 0 / 0 for t = T
        steps = torch.arange(self.num_timesteps + 1, device=x.device)
        in_range = (steps[None] > s[:, None]) & (steps[None] <= t[:, None])
        k = torch.where(in_range, self.beta[None], torch.ones_like(self.beta[None])).prod(-1)  # ty:ignore[not-subscriptable]
        k = self._expand(k, x)
        return k * x + 0.5 * (1 - k)

    def posterior(
        self,
        x0: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Probability q(x_s = 1 | x_t, x_0) of shape (B, ...) for (soft) x_0 and binary x_t.

        Parameters
        ----------
        x0 : torch.Tensor
            Probabilities of x_0 = 1, of shape (B, ...).
        xt : torch.Tensor
            Binary x_t of shape (B, ...).
        t : torch.Tensor
            Diffusion steps of shape (B,), in [1, num_timesteps].
        s : torch.Tensor or None, optional
            Earlier steps of shape (B,), s < t, by default t - 1.
        """
        if s is None:
            s = t - 1
        prior = self(torch.stack([x0, 1 - x0], dim=-1), s)
        likelihood = self.jump(torch.stack([xt, 1 - xt], dim=-1), t, s)
        probs = prior * likelihood
        probs = probs / (probs.sum(-1, keepdim=True) + 1e-6)
        return probs[..., 0]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        k = self._expand(self.k_final[t], x)  # ty:ignore[not-subscriptable]
        b = self._expand(self.b_final[t], x)  # ty:ignore[not-subscriptable]
        return k * x + b
