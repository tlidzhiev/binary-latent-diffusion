import random
from collections.abc import Callable
from typing import Any

import safetensors.torch
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index: list[dict[str, Any]],
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        index : list[dict[str, Any]]
            List containing dict for each element of the dataset.
            The dict has required metadata information, such as label and object path.
        limit : int or None, optional
            If not None, limit the total number of elements in the dataset
            to 'limit' elements, by default None.
        shuffle_index : bool, optional
            If True, shuffle the index using Python random package with seed 42,
            by default False.
        instance_transforms : dict[str, Callable] or None, optional
            Transforms that should be applied on the instance. Depend on the
            tensor name, by default None.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: list[dict[str, Any]] = index

        self.instance_transforms = instance_transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get element from the index, preprocess it, and combine it into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Parameters
        ----------
        index : int
            Index in the self._index list.

        Returns
        -------
        dict[str, Any]
            Dict containing instance (a single dataset element).
        """
        data_dict = self._index[index]
        data_path = data_dict['path']
        data_object = self.load_object(data_path)
        instance_data = {'img': data_object, 'label': data_dict['label']}
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self) -> int:
        """
        Get length of the dataset (length of the index).

        Returns
        -------
        int
            Number of elements in the dataset.
        """
        return len(self._index)

    def load_object(self, path: str) -> torch.Tensor:
        """
        Load object from disk.

        Parameters
        ----------
        path : str
            Path to the object.

        Returns
        -------
        torch.Tensor
            Tensor object.
        """
        raw = safetensors.torch.load_file(path)['img']
        return raw

    def preprocess_data(self, instance_data: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Parameters
        ----------
        instance_data : dict[str, Any]
            Dict containing instance (a single dataset element).

        Returns
        -------
        dict[str, Any]
            Dict containing instance (a single dataset element),
            possibly transformed via instance transform.
        """
        if self.instance_transforms is not None:
            for name, transform in self.instance_transforms.items():
                if name in instance_data:
                    instance_data[name] = transform(instance_data[name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Filter some of the elements from the dataset depending on some condition.

        This is not used now. The method should be called in
        the __init__ before shuffling and limiting.

        Parameters
        ----------
        index : list[dict[str, Any]]
            List containing dict for each element of the dataset.
            The dict has required metadata information, such as label and object path.

        Returns
        -------
        list[dict[str, Any]]
            List containing dict for each element of the dataset that satisfied
            the condition. The dict has required metadata information,
            such as label and object path.
        """
        # Filter logic
        return index

    @staticmethod
    def _assert_index_is_valid(index: list[dict[str, Any]]) -> None:
        """
        Check the structure of the index and ensure it satisfies the desired conditions.

        Parameters
        ----------
        index : list[dict[str, Any]]
            List containing dict for each element of the dataset.
            The dict has required metadata information, such as label and object path.

        Raises
        ------
        AssertionError
            If required fields are missing from index entries.
        """
        for entry in index:
            assert 'path' in entry, (
                "Each dataset item should include field 'path' - path to sample file."
            )

    @staticmethod
    def _sort_index(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sort index via some rules.

        This is not used now. The method should be called in
        the __init__ before shuffling and limiting and after filtering.

        Parameters
        ----------
        index : list[dict[str, Any]]
            List containing dict for each element of the dataset.
            The dict has required metadata information, such as label and object path.

        Returns
        -------
        list[dict[str, Any]]
            Sorted list containing dict for each element of the dataset.
            The dict has required metadata information, such as label and object path.
        """
        # Sorting logic
        return index

    @staticmethod
    def _shuffle_and_limit_index(
        index: list[dict[str, Any]],
        limit: int | None,
        shuffle_index: bool,
    ) -> list[dict[str, Any]]:
        """
        Shuffle elements in index and limit the total number of elements.

        Parameters
        ----------
        index : list[dict[str, Any]]
            List containing dict for each element of the dataset.
            The dict has required metadata information, such as label and object path.
        limit : int or None
            If not None, limit the total number of elements in the dataset
            to 'limit' elements.
        shuffle_index : bool
            If True, shuffle the index using Python random package with seed 42.

        Returns
        -------
        list[dict[str, Any]]
            Shuffled and limited index.
        """
        if shuffle_index:
            # Use a local Random instance so this doesn't reset the global
            # RNG state (and clobber the configured trainer/inferencer seed).
            random.Random(42).shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
