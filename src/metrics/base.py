class BaseMetric:
    """
    Base class for all metrics.
    """

    def __init__(self, name: str | None = None, *args, **kwargs):
        """
        Parameters
        ----------
        name : str or None, optional
            Metric name to use in logger and writer, by default None.
            If None, uses class name.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """

        self.name = name if name is not None else type(self).__name__

    def update(self, **kwargs):
        """
        Update metric state with batch data.

        Parameters
        ----------
        **kwargs
            Batch data containing inputs and targets.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement update method.')

    def compute(self) -> float:
        """
        Compute the metric over all batches seen since the last reset.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement compute method.')

    def reset(self) -> None:
        """
        Reset the accumulated metric state.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement reset method.')

    def __call__(self, **kwargs) -> float:
        """
        Defines metric calculation logic for a given batch.

        Can use external functions (like TorchMetrics) or custom ones.

        Parameters
        ----------
        **kwargs
            Batch data containing inputs and targets.

        Returns
        -------
        float
            Computed metric value.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclass.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement __call__ method.')

    def __str__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation.
        """
        return f'{type(self).__name__}()'

    def __repr__(self) -> str:
        """
        Return string representation of the metric.

        Returns
        -------
        str
            String representation.
        """
        return self.__str__()
