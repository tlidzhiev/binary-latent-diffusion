from collections.abc import Generator
from itertools import repeat

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, IterableDataset


def inf_loop(dataloader: DataLoader) -> Generator:
    """
    Wrapper function for endless dataloader.

    Used for iteration-based training scheme.

    Parameters
    ----------
    dataloader : DataLoader
        Classic finite dataloader.

    Yields
    ------
    Any
        Batch from the dataloader, cycling infinitely.
    """
    for loader in repeat(dataloader):
        yield from loader


def _to_device(
    batch_transforms: dict[str, dict[str, nn.Sequential]],
    device: str,
) -> None:
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Parameters
    ----------
    batch_transforms : dict[str, dict[str, nn.Sequential]]
        Transforms that should be applied on the whole batch.
        Depend on the tensor name.
    device : str
        Device to use for batch transforms.
    """
    for transforms in batch_transforms.values():
        if transforms is not None:
            for tensor_name, transform in transforms.items():
                transforms[tensor_name] = transform.to(device)


def get_dataloaders(
    config: DictConfig, device: str
) -> tuple[
    dict[str, DataLoader],
    dict[str, dict[str, nn.Sequential]],
]:
    """
    Create dataloaders for each of the dataset partitions.

    Also creates instance and batch transforms.

    Parameters
    ----------
    config : DictConfig
        Hydra experiment config.
    device : str
        Device to use for batch transforms.

    Returns
    -------
    dataloaders : dict[str, DataLoader]
        Dict containing dataloader for a partition defined by key.
    batch_transforms : dict[str, dict[str, nn.Sequential]]
        Transforms that should be applied on the whole batch.
        Depend on the tensor name.
    """

    batch_transforms = instantiate(config.transforms.batch_transforms, device)
    _to_device(batch_transforms, device)
    datasets = instantiate(config.dataset)

    dataloaders = {}
    for partition in config.dataset:
        if partition not in ['train', 'val', 'test'] or config.dataloader.get(partition) is None:
            continue

        dataset = datasets[partition]

        if not isinstance(dataset, IterableDataset):
            batch_size = config.dataloader[partition].batch_size
            assert batch_size <= len(dataset), (
                f'The batch size ({batch_size}) cannot '
                f'be larger than the dataset length ({len(dataset)})'
            )

        partition_dataloader = instantiate(config.dataloader[partition], dataset=dataset)
        dataloaders[partition] = partition_dataloader

    return dataloaders, batch_transforms
