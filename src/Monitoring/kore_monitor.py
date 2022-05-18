import os
import wandb
from src.Monitoring.missing_entity_exception import MissingEntityException

WANDB_PROJECT_NAME: str = "rl-dl-lab"
ENTITY_NAME_ENV_NAME: str = "WANDB_ENTITY"
ENTITY_NAME: str = os.environ.get(ENTITY_NAME_ENV_NAME)


class KoreMonitor:

    def __init__(self, agent_name: str, value_name: str):
        if ENTITY_NAME is None:
            raise MissingEntityException(
                f"No entity name found for WANDB. Please set your environment variable: {ENTITY_NAME_ENV_NAME}"
            )
        self.agent_name = agent_name
        self.value_name = value_name
        wandb.init(project=WANDB_PROJECT_NAME, name=self.agent_name, entity=ENTITY_NAME)

    def log_value(self, value):
        wandb.log({self.value_name: value})


def main():
    kore_monitor = KoreMonitor(agent_name="dummy_agent", value_name="dummy_value")
    for i in range(100):
        kore_monitor.log_value(i * 100)


if __name__ == "__main__":
    main()
