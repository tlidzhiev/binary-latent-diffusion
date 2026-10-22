from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel

from src.logger.base import BaseWriter


@torch.no_grad()
def get_grad_norm(model: nn.Module, norm_type: float | str | None = 2) -> float:
    """
    Calculates the gradient norm for logging.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    norm_type : float or str or None, optional
        The order of the norm, by default 2.

    Returns
    -------
    float
        The calculated norm.
    """
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]

    if len(gradients) == 0:
        return 0.0

    total_norm = torch.norm(
        torch.stack([torch.norm(grad, norm_type) for grad in gradients]),
        norm_type,
    )
    return total_norm.item()


def clip_grad_norm(model: nn.Module, max_grad_norm: float | None = None) -> None:
    """
    Clips the gradient norm by the value.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    max_grad_norm : float or None, optional
        Maximum gradient norm value. If None, no clipping is performed,
        by default None.
    """
    if max_grad_norm is not None:
        clip_grad_norm_(model.parameters(), max_grad_norm)


def save_checkpoint(
    config: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    mnt_best: float,
    not_improved_count: int,
    filename: Path | str,
    epoch: int,
    writer: BaseWriter | None = None,
    ema: AveragedModel | None = None,
) -> None:
    """
    Save checkpoint with model, optimizer, and lr_scheduler states.

    Parameters
    ----------
    config : DictConfig
        Experiment config containing training config.
    model : nn.Module
        PyTorch model.
    optimizer : Optimizer
        Optimizer for the model.
    lr_scheduler : LRScheduler
        Learning rate scheduler for the optimizer.
    mnt_best : float
        Best value of the monitored metric achieved so far.
    not_improved_count : int
        Number of epochs without improvement of the monitored metric (for early stopping).
    filename : Path or str
        Path to save the checkpoint file.
    epoch : int
        Current epoch number.
    writer : BaseWriter or None, optional
        Writer instance for logging checkpoint, by default None.
    ema : AveragedModel or None, optional
        EMA instance to save, by default None.
    """
    filename = Path(filename)
    checkpoint_dir = filename.parent
    arch = type(model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'monitor_best': mnt_best,
        'not_improved_count': not_improved_count,
        'config': config,
    }

    if ema is not None:
        state['ema'] = ema.state_dict()

    torch.save(state, str(filename))
    if writer is not None:
        writer.add_checkpoint(str(filename), str(checkpoint_dir.parent))


def load_checkpoint(path: Path | str, device: str = 'cpu') -> dict[str, Any]:
    """
    Load checkpoint from disk.

    Parameters
    ----------
    path : Path or str
        Path to the checkpoint file.
    device : str, optional
        Device to map the checkpoint tensors to, by default 'cpu'.

    Returns
    -------
    dict[str, Any]
        Dictionary containing checkpoint data including model state,
        optimizer state, epoch, and configuration.

    Warning
    -------
    Loads with 'weights_only=False' because the checkpoint embeds a
    DictConfig (not just tensors), which requires unpickling arbitrary
    objects. Only load checkpoints from a source you trust -- this is
    equivalent to running arbitrary code from the file.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def resume_checkpoint(
    config: DictConfig,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    path: Path | str,
    device: str = 'cpu',
    ema: AveragedModel | None = None,
) -> tuple[nn.Module, Optimizer, LRScheduler, int, float, int, AveragedModel | None]:
    """
    Resume from a saved checkpoint (in case of server crash, etc.).

    The function loads state dicts for everything, including model,
    optimizers, etc.

    Parameters
    ----------
    config : DictConfig
        Experiment config containing training config.
    model : nn.Module
        PyTorch model to load the state into.
    optimizer : Optimizer
        Optimizer to load the state into.
    lr_scheduler : LRScheduler
        Learning rate scheduler to load the state into.
    path : Path or str
        Path to the checkpoint to be resumed.
    device : str, optional
        Device to map the checkpoint tensors to, by default 'cpu'.
    ema : AveragedModel or None, optional
        EMA instance to save, by default None.

    Returns
    -------
    model : nn.Module
        The model with restored state dictionary.
    optimizer : Optimizer
        The optimizer with restored state dictionary.
    lr_scheduler : LRScheduler
        The learning rate scheduler with restored state dictionary.
    start_epoch : int
        The epoch number to start training from.
    mnt_best : float
        The best value of the monitored metric recorded in the checkpoint.
    not_improved_count : int
        Number of epochs without improvement recorded in the checkpoint (0 if absent).
    ema : AveragedModel or None
        The EMA instance with restored state dictionary, if provided.
    """
    path = str(path)
    checkpoint = load_checkpoint(path, device)
    start_epoch = checkpoint['epoch'] + 1
    mnt_best = cast(float, checkpoint.get('monitor_best'))
    not_improved_count = checkpoint.get('not_improved_count', 0)

    # load architecture params from checkpoint.
    if checkpoint['config']['model'] != config['model']:
        raise RuntimeError(
            'Architecture configuration given in the config file is different from that of the checkpoint.'
        )
    else:
        model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if (
        checkpoint['config']['optimizer'] != config['optimizer']
        or checkpoint['config']['lr_scheduler'] != config['lr_scheduler']
    ):
        raise RuntimeError(
            'Optimizer or lr_scheduler given in the config file is different '
            'from that of the checkpoint. Cannot resume optimizer and scheduler states.'
        )
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # load EMA state if an EMA object is passed and the state exists in the checkpoint
    if ema is not None and 'ema' in checkpoint:
        ema.load_state_dict(checkpoint['ema'])

    return model, optimizer, lr_scheduler, start_epoch, mnt_best, not_improved_count, ema


def from_pretrained(
    model: nn.Module,
    path: Path | str,
    device: str = 'cpu',
    use_ema: bool = False,
) -> nn.Module:
    """
    Init model with weights from pretrained .pth file.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    path : Path or str
        Path to the model state dict.
    device : str, optional
        Device to map the checkpoint tensors to, by default 'cpu'.
    use_ema : bool, optional
        If True, load the EMA averaged weights instead of the raw model
        weights. Requires the checkpoint to contain an 'ema' state (i.e.
        the model was trained with EMA enabled). By default False.

    Returns
    -------
    nn.Module
        Initialized PyTorch model.

    Raises
    ------
    RuntimeError
        If 'use_ema' is True but the checkpoint does not contain EMA state.
    """
    path = str(path)
    checkpoint = load_checkpoint(path=path, device=device)

    if use_ema:
        if 'ema' not in checkpoint:
            raise RuntimeError(
                f'use_ema=True but no EMA state found in checkpoint: {path}. '
                'The model was likely trained without EMA.'
            )
        ema_state = checkpoint['ema']
        prefix = 'module.'
        averaged_weights = {
            key[len(prefix) :]: value for key, value in ema_state.items() if key.startswith(prefix)
        }
        model.load_state_dict(averaged_weights)
    elif checkpoint.get('state_dict') is not None:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model
