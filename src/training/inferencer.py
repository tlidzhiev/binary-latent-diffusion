from logging import Logger
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from src.metrics.base import BaseMetric
from src.metrics.tracker import MetricTracker
from src.model.bae import BinaryAutoEncoder
from src.model.base import BaseModel
from src.utils.io import get_root

from .base import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model by generated images (FID), save them, etc.
    """

    def __init__(
        self,
        config: DictConfig,
        device: str,
        model: BaseModel,
        dataloaders: dict[str, DataLoader],
        metrics: dict[str, list[BaseMetric]],
        save_path: Path | str,
        batch_transforms: dict[str, dict[str, nn.Sequential]],
        skip_model_load: bool = False,
        logger: Logger | None = None,
        autoencoder: BinaryAutoEncoder | None = None,
    ) -> None:
        """
        Parameters
        ----------
        config : DictConfig
            Experiment config containing inferencer config.
        device : str
            Device for tensors and model.
        model : BaseModel
            PyTorch model.
        dataloaders : dict[str, DataLoader]
            Dataloaders for different sets of data.
        metrics : dict[str, list[BaseMetric]]
            Dict with the definition of metrics for inference
            (metrics['inference']). Each metric is an instance of
            src.metrics.base.BaseMetric.
        save_path : Path or str
            Path to the directory to save the generated images.
        batch_transforms : dict[str, dict[str, nn.Sequential]]
            Transforms that should be applied on the whole batch. Depend on the
            tensor name.
        skip_model_load : bool, optional
            If True, skip loading model weights from checkpoint, by default False.
        logger : Logger or None, optional
            Logger that logs output, by default None.
        autoencoder : BinaryAutoEncoder or None, optional
            Autoencoder decoding the sampled binary codes to images, by default None.
        """
        assert skip_model_load or config.inferencer.get('from_pretrained') is not None, (
            'Provide checkpoint or set skip_model_load=True'
        )

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.cfg_eval = self.config.evaluation
        self.autoencoder = autoencoder

        self.logger = logger
        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        # Path definition
        self.save_path = get_root() / save_path

        # Define metrics
        self.metrics = metrics
        self.eval_metric_tracker = MetricTracker(
            *[m.name for m in self.metrics['inference']],
        )

        if not skip_model_load:
            # Initialize model. If the checkpoint was trained with EMA and
            # inferencer.use_ema is set, load the EMA (shadow) weights instead.
            self._from_pretrained(
                config.inferencer.get('from_pretrained'),
                use_ema=config.inferencer.get('use_ema', False),
            )

    def run_inference(self) -> dict[str, Any]:
        """
        Run inference for each partition: generate images and compute the metrics
        against the real statistics of the partition.

        Returns
        -------
        dict[str, Any]
            Dict where each key is a partition name and the value contains
            logs for that partition.
        """
        part_logs = {}
        for part in self.eval_dataloaders:
            self.is_train = False
            logs, _ = self._evaluate_part(part, save_dir=self.save_path / part)
            part_logs[part] = logs
        return part_logs
