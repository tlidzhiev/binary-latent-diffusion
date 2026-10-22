import logging
import os
import random
import secrets
import shutil
import string
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.logger import setup_logging

from .io import get_root


def set_worker_seed(worker_id) -> None:
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for model training or inference.

    Parameters:
    seed : int
        Defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 run id for the experiment tracker.

    There are ~2.8T base-36 ids of length 8: generating 210k of them gives
    a ~1% chance of collision.

    See https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py.

    Parameters
    ----------
    length : int, optional
        Length of the id, by default 8.

    Returns
    -------
    str
        Random string of lowercase letters and digits.
    """
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(log_dir: Path) -> None:
    """
    Save the current git commit and uncommitted changes to the run directory.

    The code version of the run is restored with:
        git checkout <hash from git_commit.txt>
        git apply git_diff.patch

    New files get into the patch only if they are staged via 'git add'
    before the run.

    Parameters
    ----------
    log_dir : Path
        Run directory.
    """
    commands = {
        'git_commit.txt': ['git', 'rev-parse', 'HEAD'],
        'git_diff.patch': ['git', 'diff', 'HEAD'],
    }
    for filename, command in commands.items():
        # Bytes are kept as is, so the patch still applies to files with CRLF or non-UTF-8 text.
        result = subprocess.run(command, cwd=get_root(), capture_output=True, check=False)
        if result.returncode == 0:
            (log_dir / filename).write_bytes(result.stdout)


def saving_init(log_dir: Path, cfg: DictConfig) -> None:
    """
    Create the run directory and save the config into it.

    On resume the run id is taken from the saved config, so the experiment
    tracker continues the same run.

    Parameters
    ----------
    log_dir : Path
        Run directory for everything: logs, checkpoints, config, etc.
    cfg : DictConfig
        Project config. 'writer.run_id' is filled in place.

    Raises
    ------
    ValueError
        If the run directory holds a run and neither resume nor override is set.
    """
    run_id = None
    # Hydra creates the run directory and its own log before the job starts,
    # so a previous run is recognized by its saved config.
    if (log_dir / 'config.yaml').exists():
        if cfg.trainer.resume_from is not None:
            run_id = OmegaConf.load(log_dir / 'config.yaml').writer.run_id
            print(f'Resuming training from run {run_id}...')
        elif cfg.trainer.override:
            print(f"Overriding log directory '{log_dir}'...")
            shutil.rmtree(log_dir)
        else:
            raise ValueError(
                f"Log directory '{log_dir}' exists. "
                'Change writer.run_name, set trainer.override=True or trainer.resume_from.'
            )
    elif cfg.trainer.resume_from is not None:
        raise ValueError(f"Cannot resume: log directory '{log_dir}' has no run.")

    log_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.update(cfg, 'writer.run_id', run_id or generate_id(cfg.writer.id_length))
    OmegaConf.save(cfg, log_dir / 'config.yaml')
    log_git_commit_and_patch(log_dir)


def setup_saving_and_logging(cfg: DictConfig) -> logging.Logger:
    """
    Initialize the run directory and logging.

    The run directory is 'writer.log_dir' relative to the project root.
    The checkpoint directory 'trainer.checkpoint_dir' is created inside it.
    On resume the log file is appended to instead of overwritten.

    Parameters
    ----------
    cfg : DictConfig
        Config for the current experiment.

    Returns
    -------
    logging.Logger
        Logger of the training script.
    """
    log_dir = get_root() / cfg.writer.log_dir
    saving_init(log_dir, cfg)
    (get_root() / cfg.trainer.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    setup_logging(log_dir, append=cfg.trainer.resume_from is not None)
    return logging.getLogger('train')
