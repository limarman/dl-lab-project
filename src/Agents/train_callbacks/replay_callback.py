import wandb
from stable_baselines3.common.callbacks import BaseCallback


class ReplayCallback(BaseCallback):
    def __init__(self, episodes_interval=None, folder_name='default', enemy_agent='balanced'):
        super().__init__()
        self.episodes_interval = episodes_interval
        self.folder_name = folder_name
        self.enemy_agent = enemy_agent
        self.num_episodes_done = 0

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                self.num_episodes_done += 1
                if self.num_episodes_done % self.episodes_interval == 0:
                    replay = self.training_env.get_attr("replay", i)[0]
                    wandb.log({"replay": wandb.Html(replay, inject=False)})

        return True

    def _on_rollout_end(self) -> None:
        pass

