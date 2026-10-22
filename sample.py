import logging

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from src.logger import setup_logging
from src.training.utils import from_pretrained
from src.utils.init import set_random_seed
from src.utils.io import get_root


@hydra.main(version_base='1.3', config_path='src/configs', config_name='sample')
def main(config: DictConfig) -> None:
    """
    Main script for image generation. Samples binary codes with the diffusion model,
    decodes them to images with the binary autoencoder and saves the images.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    """
    cfg = config.sampler
    assert cfg.from_pretrained is not None, 'Provide sampler.from_pretrained checkpoint'
    set_random_seed(cfg.seed)

    save_path = get_root() / 'data' / 'saved' / cfg.save_path
    save_path.mkdir(exist_ok=True, parents=True)
    setup_logging(save_path)
    logger = logging.getLogger('sample')

    logger.info(f'Config:\n{OmegaConf.to_yaml(config, resolve=True)}')
    if cfg.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.device
    logger.info(f'Using device: {device}')

    model = instantiate(config.model).to(device)
    from_pretrained(model, cfg.from_pretrained, device=device, use_ema=cfg.use_ema)
    model.eval()
    autoencoder = instantiate(config.autoencoder).to(device).eval()

    condition = model.denoiser.condition
    if condition == 'class':
        assert cfg.label is not None, 'Provide sampler.label for a class-conditional model'
    elif condition != 'none':
        raise NotImplementedError(f"Sampling with '{condition}' condition is not supported")

    saved = 0
    pbar = tqdm(total=cfg.num_samples, desc='Sampling')
    while saved < cfg.num_samples:
        num = min(cfg.batch_size, cfg.num_samples - saved)
        cond = torch.full((num,), cfg.label, device=device) if condition == 'class' else None
        codes = model.sample(
            num,
            cond=cond,
            temperature=cfg.temperature,
            num_steps=cfg.num_steps,
            guidance=cfg.guidance,
        )
        images = autoencoder.decode(codes).cpu()
        for i, image in enumerate(images):
            save_image(image, save_path / f'{saved + i:06d}.png')
        if saved == 0:
            save_image(make_grid(images, nrow=4), save_path / 'grid.png')
        saved += num
        pbar.update(num)
    pbar.close()
    logger.info(f'Saved {saved} images to {save_path}')


if __name__ == '__main__':
    main()
