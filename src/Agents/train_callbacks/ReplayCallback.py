import os

from stable_baselines3.common.callbacks import BaseCallback

from src.Actions.action_adapter_rule_based import ActionAdapterRuleBased
from src.Environment.kore_env import KoreEnv
from src.Rewards.penalized_dummy_reward import PenalizedDummyReward
from src.States.advanced_state import AdvancedState


class ReplayCallback(BaseCallback):
    def __init__(self, step_func, interval=None, folder_name='default', enemy_agent='balanced'):
        super().__init__()
        self.step_func = step_func
        self.interval = interval
        self.folder_name = folder_name
        self.enemy_agent = enemy_agent
        self.counter = 0

    def _on_step(self) -> bool:
        pass

    def _on_rollout_end(self) -> None:
        self.counter += 1
        if self.interval is not None and self.counter % self.interval == 0:
            self.__save_replay(self.counter)

    def __save_replay(self, episode):
        dummy_reward = PenalizedDummyReward()
        action_adapter = ActionAdapterRuleBased()
        kore_env = KoreEnv(AdvancedState, action_adapter, dummy_reward)

        kore_env.env.run([self.step_func, self.enemy_agent])
        replay_video = kore_env.render()
        self.__write_replay(replay_video, episode)

    def __write_replay(self, replay_video: str, episode: int):
        file_path = f"../../output/replays/{self.folder_name}"
        os.makedirs(file_path, exist_ok=True)
        with open(file_path + f"/replay_video_{episode}_episodes.html",
                  "w+", encoding='utf-8') as file:
            file.write(replay_video)
