import wandb

WANDB_PROJECT_NAME: str = "rl-dl-lab"


class KoreMonitor:

    def __init__(self, agent_name: str, value_name: str):
        self.agent_name = agent_name
        self.value_name = value_name
        wandb.init(project=WANDB_PROJECT_NAME, entity="dateb", name=self.agent_name)

    def update_value(self, next_value):
        wandb.log({self.value_name: next_value})


def main():
    kore_monitor = KoreMonitor(agent_name="dummy_agent", value_name="dummy_value")
    for i in range(100):
        kore_monitor.update_value(i*100)


if __name__ == "__main__":
    main()
