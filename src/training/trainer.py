from typing import Any

import numpy as np
import torch
from einops import rearrange
from torchvision.utils import make_grid

from src.metrics.tracker import MetricTracker

from .base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class.

    Defines the logic of batch processing and sample logging for the binary diffusion model.
    """

    def process_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        metric_tracker: MetricTracker,
        part: str,
    ) -> dict[str, Any]:
        """
        Run batch through the model, compute metrics, compute loss,
        and do the training step.

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Parameters
        ----------
        batch_idx : int
            Batch index.
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.
        metric_tracker : MetricTracker
            MetricTracker object that computes and aggregates the metrics.
        part : str
            Name of the partition.

        Returns
        -------
        dict[str, Any]
            Dict-based batch containing the data from the dataloader
            (possibly transformed via batch transform), model outputs, and losses.
        """
        batch = self._to_device(batch)
        batch = self._transform_batch(batch)  # Transform batch on device -- faster

        self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        batch['loss'].backward()  # Sum of all losses is always called loss
        metric_tracker.update('grad_norm', self._get_grad_norm())
        self._clip_grad_norm()
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update metrics for each loss (in case of multiple losses)
        for loss_name in self.criterion.loss_names:
            metric_tracker.update(loss_name, batch[loss_name].item())

        for met in self.metrics['train']:
            metric_tracker.update(met.name, met(**batch))

        return batch

    @torch.no_grad()
    def _log_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        epoch: int,
    ) -> None:
        """
        Log images of the batch.

        During evaluation the generated images are logged ('samples'). During training
        the first samples of the batch are decoded to images and logged as a grid with a
        row for each of: the clean code x_0, the noisy code x_t and the denoiser prediction
        of x_0 ('denoising').

        Parameters
        ----------
        batch_idx : int
            Index of the current batch.
        batch : dict[str, Any]
            Dict-based batch after going through the 'process_batch' function,
            or {'images': generated images} after evaluation.
        epoch : int
            Current epoch number.
        """
        nrow = self.cfg_eval.get('nrow', 4)
        if not self.is_train:
            self._log_image_grid('samples', batch['images'], nrow)
            return
        if self.autoencoder is None:
            return

        num = min(nrow, len(batch['x0']))
        x0, xt, logits = batch['x0'][:num], batch['xt'][:num], batch['logits'][:num]
        # with p_flip the logits are for "x_t differs from x_0": flip the sign where x_t = 1
        x0_logits = logits * (1 - 2 * xt) if self.model.p_flip else logits
        codes = torch.cat([x0, xt, (x0_logits > 0).float()])
        size = int(codes.shape[1] ** 0.5)
        images = self.autoencoder.decode(rearrange(codes, 'b (h w) c -> b c h w', h=size))
        self._log_image_grid('denoising', images, nrow=num)

    def _log_image_grid(self, name: str, images: torch.Tensor, nrow: int) -> None:
        grid = make_grid(images.cpu(), nrow=nrow)
        image: np.ndarray = grid.permute(1, 2, 0).mul(255).byte().numpy()
        self.writer.add_image(name, image)
