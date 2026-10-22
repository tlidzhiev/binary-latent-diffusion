from pathlib import Path
from typing import Any, Literal

import comet_ml
import numpy as np

from .base import BaseWriter


class CometMLWriter(BaseWriter):
    """
    Class for experiment tracking via CometML.

    See https://www.comet.com/docs/v2/.
    """

    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
        run_name: str,
        workspace: str | None = None,
        run_id: str | None = None,
        mode: Literal['online', 'offline'] = 'online',
        log_code: bool = False,
        log_graph: bool = False,
        auto_metric_logging: bool = False,
        auto_param_logging: bool = False,
        **kwargs,
    ) -> None:
        """
        API key is expected to be provided by the user in the terminal.

        Parameters
        ----------
        project_config : dict[str, Any]
            Configuration dictionary for the project.
        project_name : str
            Name of the project inside CometML.
        run_name : str
            Name of the run.
        workspace : str or None, optional
            Name of the workspace. Used if you work in a team, by default None.
        run_id : str or None, optional
            Unique identifier for the run, by default None.
        mode : {'online', 'offline'}, optional
            If 'online', log data to remote server; if 'offline', log locally,
            by default 'online'.
        log_code : bool, optional
            Whether to log source code, by default False.
        log_graph : bool, optional
            Whether to log model graph, by default False.
        auto_metric_logging : bool, optional
            Enable automatic metric logging, by default False.
        auto_param_logging : bool, optional
            Enable automatic parameter logging, by default False.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(project_config, project_name, run_id, run_name, **kwargs)

        comet_ml.login()

        resume = project_config.get('trainer', {}).get('resume_from') is not None

        if resume:
            if mode == 'offline':
                exp_class = comet_ml.ExistingOfflineExperiment
            else:
                exp_class = comet_ml.ExistingExperiment

            self.exp = exp_class(experiment_key=self.run_id)
        else:
            if mode == 'offline':
                exp_class = comet_ml.OfflineExperiment
            else:
                exp_class = comet_ml.Experiment

            self.exp = exp_class(
                project_name=project_name,
                workspace=workspace,
                experiment_key=self.run_id,
                log_code=log_code,
                log_graph=log_graph,
                auto_metric_logging=auto_metric_logging,
                auto_param_logging=auto_param_logging,
            )
            self.exp.set_name(run_name)
            self.exp.log_parameters(parameters=_nested_lists_to_str(project_config))

        self.comet_ml = comet_ml

    def add_checkpoint(self, checkpoint_path: str, save_dir: str) -> None:
        """
        Log checkpoints to the experiment tracker.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        save_dir : str
            Path to the directory where checkpoint is saved.
        """
        # For comet, save dir is not required
        # It is kept for consistency with base class
        self.exp.log_model(name='checkpoints', file_or_folder=checkpoint_path, overwrite=True)

    def add_scalar(self, name: str, value: float) -> None:
        """
        Log a scalar to the experiment tracker.

        Parameters
        ----------
        name : str
            Name of the scalar.
        value : float
            Value of the scalar.
        """
        self.exp.log_metrics(
            {self._object_name(name): value},
            step=self.step,
        )

    def add_scalars(self, values: dict[str, float]) -> None:
        """
        Log several scalars to the experiment tracker.

        Parameters
        ----------
        values : dict[str, float]
            Dict containing scalar names and values.
        """
        self.exp.log_metrics(
            {self._object_name(k): v for k, v in values.items()},
            step=self.step,
        )

    def add_image(self, name: str, image: np.ndarray | Path | str) -> None:
        """
        Log an image to the experiment tracker.

        Parameters
        ----------
        name : str
            Name of the image.
        image : np.ndarray or Path or str
            Image as numpy array or path to image file.
        """
        self.exp.log_image(image_data=image, name=self._object_name(name), step=self.step)

    def add_histogram(
        self,
        name: str,
        values: np.ndarray,
        bins: int | None = None,
    ) -> None:
        """
        Log histogram to the experiment tracker.

        Parameters
        ----------
        name : str
            Name of the histogram to use in the tracker.
        values : np.ndarray
            Array of values to calculate histogram of.
        bins : int or None, optional
            The definition of bins for the histogram, by default None.
        """
        # For comet, bins argument is not required
        # It is kept for consistency with base class
        self.exp.log_histogram_3d(values=values, name=self._object_name(name), step=self.step)

    def close(self) -> None:
        """
        Finalize and close the experiment tracker.
        """
        self.exp.end()


def _nested_lists_to_str(value: Any) -> Any:
    # Comet rejects lists of lists as parameter values, e.g. the mixture means.
    if isinstance(value, dict):
        return {key: _nested_lists_to_str(item) for key, item in value.items()}
    if isinstance(value, list) and any(isinstance(item, list | dict) for item in value):
        return str(value)
    return value
