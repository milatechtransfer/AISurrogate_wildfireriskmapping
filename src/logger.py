import os
from pathlib import Path

from comet_ml import ExistingExperiment, Experiment


class CometLogger:
    def __init__(
        self,
        project_name: str,
        workspace: str,
        experiment_name: str,
        experiment_tags: list[str] | None = None,
        api_key: str | None = None,
        previous_experiment_key: str | None = None,
    ):
        # get the api key from os vars.
        self.api_key = api_key or os.getenv("COMET_API_KEY")
        if not self.api_key:
            raise ValueError("COMET_API_KEY env. variable not set or no API key provided.")

        experiment_kwargs = dict(
            api_key=self.api_key,
            log_code=False,
            log_graph=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_histogram_tensorboard_logging=False,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            auto_output_logging="simple",
            auto_log_co2=False,
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_network=False,
            log_env_host=False,
            log_git_metadata=False,
            log_git_patch=False,
        )

        if previous_experiment_key:
            # Resuming a preempted/requeued SLURM job: continue logging into the same
            # Comet experiment instead of creating a new one.
            self.experiment = ExistingExperiment(
                previous_experiment=previous_experiment_key,
                **experiment_kwargs,
            )
        else:
            self.experiment = Experiment(
                project_name=project_name,
                workspace=workspace,
                **experiment_kwargs,
            )
            self.experiment.set_name(experiment_name)

        if experiment_tags:
            self.experiment.add_tags(experiment_tags)

        self.experiment_key: str = self.experiment.get_key()

    def log_metrics(self, metrics: dict, step: int | None = None, epoch: int | None = None):
        if epoch is not None:
            self.experiment.set_epoch(epoch)
        self.experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_params(self, params: dict):
        self.experiment.log_parameters(params)

    def log_image(
        self,
        image_path: str | os.PathLike[str],
        name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """
        Log an image file to Comet.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        if epoch is not None:
            self.experiment.set_epoch(epoch)

        self.experiment.log_image(
            image_path,
            name=name,
            step=step,
        )
