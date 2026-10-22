from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from logging import Logger
from math import ceil
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from numpy import inf
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from src.dataset.utils import inf_loop
from src.logger.base import BaseWriter
from src.loss.base import BaseLoss
from src.metrics.base import BaseMetric
from src.metrics.fid import FID
from src.metrics.tracker import MetricTracker
from src.model.bae import BinaryAutoEncoder
from src.model.base import BaseModel
from src.utils.io import get_root

from .utils import (
    clip_grad_norm,
    from_pretrained,
    get_grad_norm,
    resume_checkpoint,
    save_checkpoint,
)


def get_ema_multi_avg_fn_with_linear_warmup(decay: float = 0.999, warmup_steps: int = 0):
    @torch.no_grad()
    def ema_update(
        ema_param_list: tuple[torch.Tensor, ...] | list[torch.Tensor],
        current_param_list: tuple[torch.Tensor, ...] | list[torch.Tensor],
        num_averaged: torch.Tensor | int,
    ) -> None:
        d = decay
        if warmup_steps > 0:
            n = num_averaged.item() if isinstance(num_averaged, torch.Tensor) else num_averaged
            if n < warmup_steps:
                d = decay * (n / warmup_steps)

        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(ema_param_list[0]):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - d)
        else:
            for p_ema, p_model in zip(ema_param_list, current_param_list, strict=True):
                p_ema.copy_(p_model)

    return ema_update


class BaseTrainer:
    """
    Base class for all trainers.

    Provides common training infrastructure including epoch management,
    metric tracking, checkpointing, and evaluation logic.
    """

    def __init__(
        self,
        config: DictConfig,
        device: str,
        dataloaders: dict[str, DataLoader],
        model: BaseModel,
        criterion: BaseLoss,
        metrics: dict[str, list[BaseMetric]],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        writer: BaseWriter,
        batch_transforms: dict[str, dict[str, nn.Sequential]],
        skip_oom: bool = True,
        epoch_len: int | None = None,
        logger: Logger | None = None,
        autoencoder: BinaryAutoEncoder | None = None,
    ) -> None:
        """
        Parameters
        ----------
        config : DictConfig
            Experiment config containing training config.
        device : str
            Device for tensors and model.
        dataloaders : dict[str, DataLoader]
            Dataloaders for different sets of data.
        model : BaseModel
            PyTorch model.
        criterion : BaseLoss
            Loss function for model training.
        metrics : dict[str, list[BaseMetric]]
            Dict with the definition of metrics for training
            (metrics['train']) and inference (metrics['inference']). Each
            metric is an instance of src.metrics.base.BaseMetric.
        optimizer : Optimizer
            Optimizer for the model.
        lr_scheduler : LRScheduler
            Learning rate scheduler for the optimizer.
        writer : BaseWriter
            Experiment tracker.
        batch_transforms : dict[str, dict[str, nn.Sequential]]
            Transforms that should be applied on the whole batch. Depend on the
            tensor name.
        skip_oom : bool, optional
            Skip batches with the OutOfMemory error, by default True.
        epoch_len : int or None, optional
            Number of steps in each epoch for iteration-based training.
            If None, use epoch-based training (len(dataloader)), by default None.
        logger : Logger or None, optional
            Logger that logs output, by default None.
        autoencoder : BinaryAutoEncoder or None, optional
            Autoencoder decoding the sampled binary codes to images, by default None.
            Required for evaluation.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer
        self.cfg_eval = self.config.evaluation
        self.autoencoder = autoencoder

        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = self.cfg_trainer.get('log_step', 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # Initialize EMA if configured
        self.ema: AveragedModel | None = None
        if (ema_cfg := self.cfg_trainer.get('ema')) is not None:
            ema_decay = ema_cfg.get('decay', 0.999)
            ema_warmup_steps = ema_cfg.get('warmup_steps', 0)
            self.ema = AveragedModel(
                model,
                multi_avg_fn=get_ema_multi_avg_fn_with_linear_warmup(
                    decay=ema_decay,
                    warmup_steps=ema_warmup_steps,
                ),
                use_buffers=True,
            )
            self._log(f'EMA enabled with decay={ema_decay}, warmup_steps={ema_warmup_steps}')

        # Define metrics
        self.metrics = metrics
        self.train_metric_tracker = MetricTracker(
            *self.criterion.loss_names,
            'grad_norm',
            *[m.name for m in self.metrics['train']],
        )
        self.eval_metric_tracker = MetricTracker(*[m.name for m in self.metrics['inference']])

        # Define dataloaders
        if epoch_len is None:
            self.train_dataloader = dataloaders['train']
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(dataloaders['train'])
            self.epoch_len = epoch_len
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        # Define epochs
        self._last_epoch = 0  # Last fully completed epoch, required for saving on interruption
        self.start_epoch = 1
        self.num_epochs = self.cfg_trainer.num_epochs

        # Configuration to monitor model performance and save best
        self.save_period = self.cfg_trainer.save_period  # Checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get('monitor', 'off')  # Format: "mnt_mode mnt_metric"
        self.only_best = self.cfg_trainer.get('only_best', False)  # Save only model-best.pth
        if self.only_best and self.monitor == 'off':
            raise ValueError("trainer.only_best requires trainer.monitor to be set (not 'off').")

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)
            if self.early_stop is None or self.early_stop <= 0:
                self.early_stop = inf

        # Epochs without improvement, saved in checkpoints to keep early stopping on resume
        self.not_improved_count = 0

        # Setup visualization writer instance
        self.writer = writer

        # Define checkpoint dir and init everything if required
        self.checkpoint_dir: Path = get_root() / self.cfg_trainer.checkpoint_dir

        if self.cfg_trainer.get('resume_from') is not None:
            resume_path = self.checkpoint_dir / self.cfg_trainer.resume_from
            self._resume_checkpoint(resume_path)

        if self.cfg_trainer.get('from_pretrained') is not None:
            self._from_pretrained(self.cfg_trainer.from_pretrained)

    def train(self) -> None:
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt:
            self._log('Saving model on keyboard interrupt.')
            self._save_checkpoint(self._last_epoch, interrupted=True)
            self.writer.close()
            raise

    def _train_process(self) -> None:
        """
        Full training logic.

        Trains model for multiple epochs, evaluates it on non-train partitions,
        and monitors the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            result = self._train_epoch(epoch)
            self._last_epoch = epoch

            # Save logged information into logs dict
            logs = {'epoch': epoch}
            logs.update(result)

            # Print logged information to the screen
            self._log(f'\nMetrics:\n{yaml.dump(logs)}')

            # Evaluate model performance according to configured metric,
            # Save best checkpoint as model_best
            best, stop_process, self.not_improved_count = self._monitor_performance(
                logs, self.not_improved_count
            )

            # Periodic and last-epoch checkpoints are saved in addition to the best one
            if not self.only_best and (epoch % self.save_period == 0 or epoch == self.num_epochs):
                self._save_checkpoint(epoch, save_best=False)
            if best:
                self._save_checkpoint(epoch, save_best=True)

            if stop_process:  # Early_stop
                break

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        dict[str, Any]
            Logs that contain the average loss and metric in this epoch.
        """
        self.is_train = True
        self.model.train()
        self.train_metric_tracker.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar('epoch', epoch)

        last_train_metrics: dict[str, Any] = {}
        logged_up_to_date = True

        pbar = tqdm(self.train_dataloader, desc=f'Train Epoch {epoch}', total=self.epoch_len)
        for batch_idx, batch in enumerate(pbar):
            try:
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    metric_tracker=self.train_metric_tracker,
                    part='train',
                )
            except torch.cuda.OutOfMemoryError:
                if self.skip_oom:
                    self._log('OOM on batch. Skipping batch.', message_type='WARNING')
                    torch.cuda.empty_cache()  # Free some memory
                    continue
                else:
                    raise

            # Update EMA after optimizer step
            if self.ema is not None:
                self.ema.update_parameters(self.model)

            # Log current results
            if batch_idx > 0 and batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.writer.add_scalar('lr', float(self.lr_scheduler.get_last_lr()[0]))
                # We don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metric_tracker.result()
                self.train_metric_tracker.reset()
                logged_up_to_date = True
                self.writer.add_scalars(last_train_metrics)
                self._log_batch(batch_idx, batch, epoch)
            else:
                logged_up_to_date = False

            if batch_idx + 1 >= self.epoch_len:
                break

        if not logged_up_to_date:
            last_train_metrics = self.train_metric_tracker.result()

        logs = {f'train_{name}': value for name, value in last_train_metrics.items()}

        # Run val/test (use EMA weights if available)
        if self.ema is not None:
            self._log('Switching to EMA model...')
        for part in self.eval_dataloaders:
            with self._ema_scope():
                val_logs = self._evaluation_epoch(epoch, part)
            logs.update(**{f'{part}_{name}': value for name, value in val_logs.items()})

        return logs

    @contextmanager
    def _ema_scope(self) -> Generator[nn.Module]:
        """
        Temporarily swap 'self.model' for the EMA-averaged model.

        Yields
        ------
        nn.Module
            The model currently bound to 'self.model' (the averaged one if
            EMA is enabled, the original one otherwise).

        Notes
        -----
        No-op when EMA is not configured. There is no need to call '.eval()'
        here: '_evaluation_epoch' does it on whatever 'self.model' points to,
        and '_train_epoch' calls '.train()' on the restored original model.
        """
        if self.ema is None:
            yield self.model
            return

        original_model = self.model
        self.model = self.ema.module
        try:
            yield self.model
        finally:
            self.model = original_model

    @torch.no_grad()
    def cache_real_statistics(self, force: bool = False) -> None:
        """
        Compute and save the FID statistics of the real images of each evaluation partition.

        Passes once over the evaluation dataloaders (without batch transforms, the metrics need
        the images themselves). Existing statistics are reused, so call it before the training.

        Parameters
        ----------
        force : bool, optional
            Recompute the statistics even if they are already cached, by default False.
        """
        for part in self.eval_dataloaders:
            self._cache_real_statistics(part, force=force)

    def _stats_path(self, part: str, metric: BaseMetric) -> Path:
        return get_root() / self.cfg_eval.stats_dir / f'{part}_{metric.name}.pth'

    @torch.no_grad()
    def _cache_real_statistics(self, part: str, force: bool = False) -> None:
        fids = [m for m in self.metrics['inference'] if isinstance(m, FID)]
        fids = [m for m in fids if force or not self._stats_path(part, m).exists()]
        if not fids:
            return

        self._log(f"Computing real image statistics of '{part}' for {[m.name for m in fids]}...")
        for fid in fids:
            fid.reset_real_state()
        for batch in tqdm(self.eval_dataloaders[part], desc=f'Real statistics: {part}'):
            images = self._to_device(batch)['img']
            for fid in fids:
                fid.update_real(images)
        for fid in fids:
            fid.save_real_state(self._stats_path(part, fid))
            self._log(f'{fid.name}: {fid.num_real_samples} real images of {part}')

    def _generate_images(self, num_samples: int) -> Generator[torch.Tensor]:
        """
        Sample binary codes with the model and decode them to images, batch by batch.
        """
        assert self.autoencoder is not None, 'Evaluation requires an autoencoder'
        model: Any = self.model
        condition = model.denoiser.condition
        if condition not in ('none', 'class'):
            raise NotImplementedError(f"Sampling with '{condition}' condition is not supported")

        model.eval()
        batch_size = self.cfg_eval.batch_size
        generated = 0
        while generated < num_samples:
            num = min(batch_size, num_samples - generated)
            cond = None
            if condition == 'class':
                labels = generated + torch.arange(num, device=self.device)
                cond = labels % model.denoiser.cond_emb.num_embeddings
            codes = model.sample(
                num,
                cond=cond,
                temperature=self.cfg_eval.get('temperature', 1.0),
                num_steps=self.cfg_eval.get('num_steps'),
                guidance=self.cfg_eval.get('guidance'),
            )
            yield self.autoencoder.decode(codes)
            generated += num

    @torch.no_grad()
    def _evaluate_part(
        self,
        part: str,
        save_dir: Path | None = None,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """
        Generate images with the model and update the inference metrics with them.

        Parameters
        ----------
        part : str
            Partition whose real statistics the generated images are compared to.
        save_dir : Path or None, optional
            Directory to save all generated images to, by default None.

        Returns
        -------
        metrics : dict[str, Any]
            Values of the inference metrics.
        images : torch.Tensor
            First 'evaluation.num_logged_images' generated images.
        """
        self._cache_real_statistics(part)
        for met in self.metrics['inference']:
            if isinstance(met, FID):
                met.load_real_state(self._stats_path(part, met))
        self.eval_metric_tracker.reset()

        if save_dir is not None:
            save_dir.mkdir(exist_ok=True, parents=True)
        num_samples = self.cfg_eval.num_samples
        num_logged = self.cfg_eval.get('num_logged_images', 16)
        logged: list[torch.Tensor] = []
        saved = 0
        for images in tqdm(
            self._generate_images(num_samples),
            desc=f'Generating images: {part}',
            total=ceil(num_samples / self.cfg_eval.batch_size),
        ):
            for met in self.metrics['inference']:
                met.update(images=images)
            if sum(len(x) for x in logged) < num_logged:
                logged.append(images.cpu())
            if save_dir is not None:
                for image in images:
                    save_image(image, save_dir / f'{saved:06d}.png')
                    saved += 1

        for met in self.metrics['inference']:
            self.eval_metric_tracker.update(met.name, met.compute())
            met.reset()
        return self.eval_metric_tracker.result(), torch.cat(logged)[:num_logged]

    @torch.no_grad()
    def _evaluation_epoch(self, epoch: int, part: str) -> dict[str, Any]:
        """
        Evaluate the model after training for an epoch.

        Instead of running the model on the partition, it generates images and compares
        them to the cached statistics of the real images of the partition (FID).

        Parameters
        ----------
        epoch : int
            Current training epoch.
        part : str
            Partition whose real images the generated ones are compared to.

        Returns
        -------
        dict[str, Any]
            Logs that contain the information about evaluation.
        """
        self.is_train = False
        metrics, images = self._evaluate_part(part)

        self.writer.set_step(epoch * self.epoch_len, part)
        self.writer.add_scalars(metrics)
        self._log_batch(0, {'images': images}, epoch)
        return metrics

    def process_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        metric_tracker: MetricTracker,
        part: str,
    ) -> dict[str, Any]:
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

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
            The metrics depend on the type of the partition (train or inference).
        part : str
            Name of the partition.

        Returns
        -------
        dict[str, Any]
            Dict-based batch containing the data from the dataloader
            (possibly transformed via batch transform), model outputs, and losses.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement process_batch method.')

    def _log_batch(
        self,
        batch_idx: int,
        batch: dict[str, Any],
        epoch: int,
    ) -> None:
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Called with a training batch ('self.is_train' is True) and after each evaluation
        with the generated images, batch = {'images': images}.

        Parameters
        ----------
        batch_idx : int
            Index of the current batch.
        batch : dict[str, Any]
            Dict-based batch after going through the 'process_batch' function.
        epoch : int
            Current epoch number.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement _log_batch method.')

    def _monitor_performance(
        self,
        logs: dict[str, Any],
        not_improved_count: int,
    ) -> tuple[bool, bool, int]:
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Parameters
        ----------
        logs : dict[str, float]
            Logs after training and evaluating the model for an epoch.
        not_improved_count : int
            The current number of epochs without improvement.

        Returns
        -------
        best : bool
            If True, the monitored metric has improved.
        stop_process : bool
            If True, stop the process (early stopping).
            The metric did not improve for too many epochs.
        not_improved_count : int
            Updated number of epochs without improvement.
        """
        if self.mnt_mode == 'off':
            return False, False, not_improved_count

        if self.mnt_metric not in logs:
            self._log(
                f"Metric '{self.mnt_metric}' is not found. Performance monitoring is disabled.",
                message_type='WARNING',
            )
            self.mnt_mode = 'off'
            return False, False, not_improved_count

        metric = float(logs[self.mnt_metric])
        improved = metric < self.mnt_best if self.mnt_mode == 'min' else metric > self.mnt_best
        if improved:
            self.mnt_best = metric
            return True, False, 0

        not_improved_count += 1
        stop_process = not_improved_count >= self.early_stop
        if stop_process:
            self._log(
                f"Validation performance didn't improve for {self.early_stop} evaluations. "
                'Training stops.'
            )
        return False, stop_process, not_improved_count

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Move all necessary tensors to the device.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.

        Returns
        -------
        dict[str, Any]
            Dict-based batch containing the data from the dataloader
            with some of the tensors on the device.
        """
        for key in self.cfg_trainer.device_tensors:
            if key in batch:
                batch[key] = batch[key].to(self.device)
        return batch

    def _transform_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Parameters
        ----------
        batch : dict[str, Any]
            Dict-based batch containing the data from the dataloader.

        Returns
        -------
        dict[str, Any]
            Dict-based batch containing the data from the dataloader
            (possibly transformed via batch transform).
        """
        transform_type = 'train' if self.is_train else 'inference'
        transforms = self.batch_transforms.get(transform_type)

        if transforms is None:
            return batch

        for transform_name, transform_fn in transforms.items():
            batch[transform_name] = transform_fn(batch[transform_name])
        return batch

    def _clip_grad_norm(self) -> None:
        """
        Clips the gradient norm by the value defined in config.trainer.max_grad_norm.
        """
        clip_grad_norm(model=self.model, max_grad_norm=self.cfg_trainer.get('max_grad_norm'))

    @torch.no_grad()
    def _get_grad_norm(self, norm_type: float | str | None = 2) -> float:
        """
        Calculates the gradient norm for logging.

        Parameters
        ----------
        norm_type : float or str or None, optional
            The order of the norm, by default 2.

        Returns
        -------
        float
            The calculated norm.
        """
        total_grad_norm = get_grad_norm(model=self.model, norm_type=norm_type)
        return total_grad_norm

    def _save_checkpoint(
        self,
        epoch: int,
        save_best: bool = False,
        interrupted: bool = False,
    ) -> None:
        """
        Save the checkpoints.

        Parameters
        ----------
        epoch : int
            Last fully completed epoch number.
        save_best : bool, optional
            If True, save the checkpoint as 'model-best.pth', by default False.
        interrupted : bool, optional
            If True, save the checkpoint as 'checkpoint-interrupted.pth'
            (training was interrupted in the middle of the epoch 'epoch + 1'),
            by default False.
        """
        if interrupted:
            filename = str(self.checkpoint_dir / 'checkpoint-interrupted.pth')
            self._log(f'Saving interrupted checkpoint: {filename}...')
        elif save_best:
            filename = str(self.checkpoint_dir / 'model-best.pth')
            self._log(f'Saving current best: {filename}...')
        else:
            filename = str(self.checkpoint_dir / f'checkpoint-epoch-{epoch}.pth')
            self._log(f'Saving checkpoint: {filename}...')

        save_checkpoint(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            mnt_best=self.mnt_best,
            not_improved_count=self.not_improved_count,
            filename=filename,
            epoch=epoch,
            writer=self.writer,
            ema=self.ema,
        )

    def _resume_checkpoint(self, resume_path: Path | str) -> None:
        """
        Resume from a saved checkpoint (in case of server crash, etc.).

        The function loads state dicts for everything, including model,
        optimizers, etc.

        Parameters
        ----------
        resume_path : Path or str
            Path to the checkpoint to be resumed.

        Notes
        -----
        The checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').
        """
        model, optimizer, lr_scheduler, start_epoch, mnt_best, not_improved_count, ema = (
            resume_checkpoint(
                config=self.config,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                path=resume_path,
                device=self.device,
                ema=self.ema,
            )
        )
        self.model = model
        self.ema = ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.start_epoch = start_epoch
        self.mnt_best = mnt_best
        self.not_improved_count = not_improved_count

        self._log(f'Checkpoint loaded. Resume training from epoch {self.start_epoch - 1}')

    def _from_pretrained(self, pretrained_path: Path | str, use_ema: bool = False) -> None:
        """
        Init model with weights from pretrained pth file.

        Parameters
        ----------
        pretrained_path : Path or str
            Path to the model state dict.
        use_ema : bool, optional
            If True, load the EMA weights from the checkpoint instead
            of the raw model weights. By default False.

        Notes
        -----
        'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.
        """
        source = 'EMA weights' if use_ema else 'model weights'
        self._log(f'Loading {source} from: {pretrained_path}...')
        self.model = from_pretrained(
            model=self.model,
            path=pretrained_path,
            device=self.device,
            use_ema=use_ema,
        )

    def _log(
        self,
        message: str,
        message_type: Literal['INFO', 'WARNING', 'DEBUG'] = 'INFO',
    ) -> None:
        """
        Log a message using the configured logger.

        Parameters
        ----------
        message : str
            Message to log.
        message_type : {'INFO', 'WARNING', 'DEBUG'}, optional
            Type of the log message, by default 'INFO'.
        """
        message = f'{type(self).__name__}: {message}'
        if self.logger is not None:
            match message_type:
                case 'INFO':
                    self.logger.info(message)
                case 'DEBUG':
                    self.logger.debug(message)
                case 'WARNING':
                    self.logger.warning(message)
        else:
            print(f'{datetime.now(UTC).astimezone()} {message_type}: {message}')
