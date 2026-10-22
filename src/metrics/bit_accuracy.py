import torch

from .base import BaseMetric


class BitAccuracy(BaseMetric):
    """
    Accuracy of the denoiser in predicting bits of the clean binary code.
    """

    def __init__(self, p_flip: bool = True, name: str | None = None) -> None:
        """
        Parameters
        ----------
        p_flip : bool, optional
            Whether the denoiser predicts bit flips of x_t instead of x_0, by default True.
            Must match the criterion.
        name : str or None, optional
            Metric name to use in logger and writer, by default None.
        """
        super().__init__(name)
        self.p_flip = p_flip
        self.reset()

    def reset(self) -> None:
        self._correct = 0.0
        self._total = 0

    @torch.no_grad()
    def _count(self, logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor) -> tuple[float, int]:
        # with p_flip the logits are for "x_t differs from x_0": flip the sign where x_t = 1
        x0_logits = logits * (1 - 2 * xt) if self.p_flip else logits
        correct = ((x0_logits > 0).float() == x0).sum().item()
        return correct, x0.numel()

    def update(self, logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, **kwargs) -> None:
        correct, total = self._count(logits, x0, xt)
        self._correct += correct
        self._total += total

    def compute(self) -> float:
        return self._correct / max(1, self._total)

    def __call__(self, logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, **kwargs) -> float:
        correct, total = self._count(logits, x0, xt)
        return correct / total
