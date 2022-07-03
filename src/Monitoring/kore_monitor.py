import os
import wandb
from wandb.integration.sb3 import WandbCallback

from src.Monitoring.missing_entity_exception import MissingEntityException

WANDB_PROJECT_NAME: str = "rl-dl-lab"
ENTITY_NAME_ENV_NAME: str = "WANDB_ENTITY"
ENTITY_NAME: str = os.environ.get(ENTITY_NAME_ENV_NAME)


class KoreMonitor:

    def __init__(self, run_id: str, resume_training=False):
        if ENTITY_NAME is None:
            raise MissingEntityException(
                f"No entity name found for WANDB. Please set your environment variable: {ENTITY_NAME_ENV_NAME}"
            )

        entity = os.environ.get("WANDB_ENTITY")
        self.run = wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=entity,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            resume=resume_training,
            id=run_id

        )

        self.callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"runs/{self.run.id}",
            verbose=2
        )

        self.tensorboard_log = f"runs/{self.run.id}"

    def set_run_name(self, name: str):
        self.run.name = name
