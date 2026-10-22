from pathlib import Path

import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance

from .base import BaseMetric

_REAL_STATE_NAMES = ('real_features_sum', 'real_features_cov_sum', 'real_features_num_samples')


class FID(BaseMetric):
    """
    Frechet Inception Distance with cached statistics of the real images.

    Wraps `torchmetrics.image.fid.FrechetInceptionDistance`. It accumulates the sum and the
    sum of outer products of the Inception features, so the statistics of the real images
    take 2048 * (2048 + 1) float64 numbers (32 MB) and can be computed once and reused:
    `update_real` + `save_real_state` once, `load_real_state` + `update` + `compute` afterwards.
    """

    def __init__(
        self,
        feature: int | nn.Module = 2048,
        device: str = 'cpu',
        name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        feature : int or nn.Module, optional
            Inception layer (64, 192, 768 or 2048) or a custom feature extractor, by default 2048.
            An integer requires `torch-fidelity` and downloads the Inception weights.
        device : str, optional
            Device of the metric, 'auto' picks cuda if available, by default 'cpu'.
        name : str or None, optional
            Metric name to use in logger and writer, by default None.
        """
        super().__init__(name)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        # the real statistics survive reset(), only the generated ones are cleared
        self.fid = FrechetInceptionDistance(
            feature=feature,
            reset_real_features=False,
            normalize=True,
        ).to(device)

    @property
    def num_real_samples(self) -> int:
        """
        Number of real images the statistics are accumulated from.
        """
        return int(self.fid.real_features_num_samples)

    def update_real(self, images: torch.Tensor, **kwargs) -> None:
        """
        Accumulate statistics of real images.

        Parameters
        ----------
        images : torch.Tensor
            Real images of shape (B, 3, H, W), float values in [0, 1].
        """
        self.fid.update(images.to(self.device), real=True)  # ty:ignore[invalid-argument-type]

    def update(self, images: torch.Tensor, **kwargs) -> None:
        """
        Accumulate statistics of generated images.

        Parameters
        ----------
        images : torch.Tensor
            Generated images of shape (B, 3, H, W), float values in [0, 1].
        """
        self.fid.update(images.to(self.device), real=False)  # ty:ignore[invalid-argument-type]

    def compute(self) -> float:
        """
        FID between the real statistics and the generated images accumulated since the last reset.

        Raises
        ------
        RuntimeError
            If there are no real statistics, call `load_real_state` or `update_real` first.
        """
        if self.num_real_samples < 2:
            raise RuntimeError(
                'No statistics of real images: call load_real_state() or update_real() first.'
            )
        return float(self.fid.compute())  # ty:ignore[missing-argument]

    def reset(self) -> None:
        """
        Reset the statistics of the generated images, the real ones are kept.
        """
        self.fid.reset()

    def reset_real_state(self) -> None:
        """
        Clear the statistics of the real images, e.g. before computing them for another dataset.
        """
        for name in _REAL_STATE_NAMES:
            setattr(self.fid, name, torch.zeros_like(getattr(self.fid, name)))

    def save_real_state(self, path: str | Path) -> None:
        """
        Save the statistics of the real images.

        Parameters
        ----------
        path : str or Path
            Path to the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({name: getattr(self.fid, name).cpu() for name in _REAL_STATE_NAMES}, path)

    def load_real_state(self, path: str | Path) -> None:
        """
        Load the statistics of the real images saved by `save_real_state`.

        The statistics are valid only for the dataset, split, preprocessing and
        feature extractor they were computed with.

        Parameters
        ----------
        path : str or Path
            Path to the file.

        Raises
        ------
        ValueError
            If the statistics were computed with a feature extractor of another size.
        """
        state = torch.load(path, map_location='cpu')
        expected = self.fid.real_features_sum.shape
        if state['real_features_sum'].shape != expected:
            raise ValueError(
                f'Statistics in {path} have {state["real_features_sum"].shape[0]} features, '
                f'the metric expects {expected[0]}.'
            )
        for name in _REAL_STATE_NAMES:
            setattr(self.fid, name, state[name].to(self.device))

    def __str__(self) -> str:
        return f'{type(self).__name__}(num_real_samples={self.num_real_samples})'
