import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.dataset.utils import get_dataloaders
from src.training.trainer import Trainer
from src.transforms.binary_encode import BinaryEncode
from src.utils.init import set_random_seed, setup_saving_and_logging
from src.utils.torch import get_lr_scheduler


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(config: DictConfig) -> None:
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    logger = setup_saving_and_logging(config)

    project_config = OmegaConf.to_container(config, resolve=True)
    writer = instantiate(config.writer, project_config)

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')
    if config.trainer.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.trainer.device
    logger.info(f'Using device: {device}')

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(f'Model:\n{model}')
    optimizer = instantiate(
        config.optimizer,
        params=[p for p in model.parameters() if p.requires_grad],
    )
    logger.info(f'Optimizer:\n{optimizer}')

    lr_scheduler = get_lr_scheduler(
        config=config,
        optimizer=optimizer,
        epoch_len=config.trainer.epoch_len
        if config.trainer.epoch_len is not None
        else len(dataloaders['train']),
    )
    logger.info(f'LR Scheduler: {lr_scheduler}')

    # the frozen autoencoder of the batch transform also decodes the sampled codes
    encode = batch_transforms['inference']['img']
    assert isinstance(encode, BinaryEncode)

    criterion = instantiate(config.criterion).to(device)
    logger.info(f'Criterion: {criterion}')
    metrics = instantiate(config.metrics)
    logger.info(f'Metrics: {metrics}')

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get('epoch_len')
    trainer = Trainer(
        config=config,
        device=device,
        dataloaders=dataloaders,
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        writer=writer,
        skip_oom=config.trainer.get('skip_oom', True),
        batch_transforms=batch_transforms,
        epoch_len=epoch_len,
        autoencoder=encode.autoencoder,
    )
    trainer.cache_real_statistics()
    trainer.train()
    writer.close()


if __name__ == '__main__':
    main()
