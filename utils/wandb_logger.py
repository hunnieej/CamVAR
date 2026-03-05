"""WandB Logger wrapper to replace Tensorboard logging."""

import wandb
from typing import Dict, Any, Optional
import os


class WandbLogger:
    """Wrapper for WandB logging compatible with existing Tensorboard API."""

    def __init__(
        self,
        project: str = "STAR-T2I-RayAdaptation",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        resume: bool = False,
        id: Optional[str] = None,
        mode: str = "online",  # "online", "offline", or "disabled"
    ):
        """Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name (experiment name)
            config: Configuration dict to log
            dir: Directory to store WandB logs
            resume: Whether to resume from previous run
            id: Run ID for resuming
            mode: WandB mode ("online", "offline", "disabled")
        """
        self.project = project
        self.name = name
        self.dir = dir or "./wandb_logs"
        self.mode = mode
        self._step = 0

        # Initialize WandB
        if resume and id:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                dir=self.dir,
                resume="must",
                id=id,
                mode=mode,
            )
        else:
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                dir=self.dir,
                mode=mode,
            )

        print(f"[WandB] Initialized: project={project}, name={name}, mode={mode}")
        print(f"[WandB] Run URL: {self.run.get_url() if self.run else 'N/A'}")

    def set_step(self, step: int):
        """Set the current global step for logging."""
        self._step = step

    def update(self, head: str = "", **kwargs):
        """Log metrics to WandB.

        Args:
            head: Metric group prefix (e.g., "AR_opt_lr")
            **kwargs: Metrics to log as key=value pairs
        """
        if not kwargs:
            return

        # Build metric dict with proper naming
        metrics = {}
        for key, value in kwargs.items():
            if head:
                metric_name = f"{head}/{key}"
            else:
                metric_name = key
            metrics[metric_name] = value

        # Log to WandB
        wandb.log(metrics, step=self._step)

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log a dictionary of metrics.

        Args:
            metrics: Dict of metrics to log
            step: Optional step override
        """
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics, step=self._step)

    def log_image(self, key: str, image, step: Optional[int] = None, caption: str = ""):
        """Log an image to WandB.

        Args:
            key: Metric name
            image: PIL Image, numpy array, or torch tensor
            step: Optional step override
            caption: Image caption
        """
        if step is None:
            step = self._step

        wandb.log({key: wandb.Image(image, caption=caption)}, step=step)

    def log_images(
        self, key: str, images: list, step: Optional[int] = None, captions: list = None
    ):
        """Log multiple images to WandB.

        Args:
            key: Metric name
            images: List of PIL Images, numpy arrays, or torch tensors
            step: Optional step override
            captions: List of captions for each image
        """
        if step is None:
            step = self._step

        if captions is None:
            captions = [""] * len(images)

        wandb_images = [
            wandb.Image(img, caption=cap) for img, cap in zip(images, captions)
        ]
        wandb.log({key: wandb_images}, step=step)

    def finish(self):
        """Finish the WandB run."""
        if self.run:
            self.run.finish()
            print("[WandB] Run finished")

    def flush(self):
        """Flush any pending logs (for compatibility with TensorboardLogger API)."""
        # WandB logs are immediately synced, so this is a no-op for compatibility
        pass

    def close(self):
        """Close the logger (alias for finish())."""
        self.finish()

    def __del__(self):
        """Cleanup when logger is deleted."""
        try:
            if hasattr(self, "run") and self.run:
                self.run.finish()
        except:
            pass


def init_wandb_logger(args, resume_id: Optional[str] = None) -> WandbLogger:
    """Initialize WandB logger from args.

    Args:
        args: Args object with configuration
        resume_id: Optional run ID for resuming

    Returns:
        WandbLogger instance
    """
    # Get project name from args (default to "STAR-T2I-RayAdaptation")
    project_name = getattr(args, "wandb_project", "STAR-T2I-RayAdaptation")

    # Get run name from args (default to exp_name)
    run_name = getattr(args, "wandb_run_name", None)
    if run_name is None:
        run_name = getattr(args, "exp_name", "experiment")

    # Build config dict from args
    config = vars(args) if hasattr(args, "__dict__") else {}

    # Filter out non-serializable objects
    config_filtered = {}
    for key, value in config.items():
        try:
            # Skip functions, modules, and other non-serializable objects
            if isinstance(
                value, (str, int, float, bool, list, dict, tuple, type(None))
            ):
                config_filtered[key] = value
        except:
            pass

    # Get output directory
    output_dir = getattr(args, "local_out_dir_path", "./outputs")
    wandb_dir = os.path.join(output_dir, "wandb_logs")

    # Get WandB mode from args (default to online)
    wandb_mode = getattr(args, "wandb_mode", "online")

    # Initialize logger
    logger = WandbLogger(
        project=project_name,
        name=run_name,
        config=config_filtered,
        dir=wandb_dir,
        resume=resume_id is not None,
        id=resume_id,
        mode=wandb_mode,
    )

    return logger
