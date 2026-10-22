import logging

import hydra
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.dataset.utils import get_dataloaders
from src.logger import setup_logging
from src.training.inferencer import Inferencer
from src.transforms.binary_encode import BinaryEncode
from src.utils.init import set_random_seed
from src.utils.io import get_root


@hydra.main(version_base='1.3', config_path='src/configs', config_name='inference')
def main(config: DictConfig) -> None:
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    save_path = get_root() / 'data' / 'saved' / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)
    setup_logging(save_path)
    logger = logging.getLogger('inference')

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')
    if config.inferencer.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.inferencer.device
    logger.info(f'Using device: {device}')

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(f'Model:\n{model}')

    metrics = instantiate(config.metrics)
    logger.info(f'Metrics: {metrics}')

    # the frozen autoencoder of the batch transform also decodes the sampled codes
    encode = batch_transforms['inference']['img']
    assert isinstance(encode, BinaryEncode)

    inferencer = Inferencer(
        config=config,
        device=device,
        dataloaders=dataloaders,
        model=model,
        metrics=metrics,
        save_path=save_path,
        batch_transforms=batch_transforms,
        skip_model_load=False,
        logger=logger,
        autoencoder=encode.autoencoder,
    )

    logs = inferencer.run_inference()
    for part, part_logs in logs.items():
        logger.info(f'\n{part} metrics:\n{yaml.dump(part_logs)}')


if __name__ == '__main__':
    main()
