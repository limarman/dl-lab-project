import os

import wandb
from kaggle_environments.envs.kore_fleets.helpers import Board
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from src.Agents.train_callbacks.ReplayCallback import ReplayCallback
from src.Environment.kore_env import KoreEnv
from src.States.board_wrapper import BoardWrapper


class A2CAgent:

    def __init__(
            self,
            kore_env: KoreEnv,
            training_steps: int = 1500000,
            name: str = 'dqg_agent',
    ):
        self.unwrapped_env = kore_env
        self.state_constr = kore_env.state_constr
        self.action_adapter = kore_env.action_adapter
        self.training_steps = training_steps
        self.name = name

        config = {
            "policy_type": "A2CMlp",
            "total_timesteps": 500000,
            "env_name": "kore_fleets",
        }
        entity = os.environ.get("WANDB_ENTITY")
        run = wandb.init(
            project="rl-dl-lab",
            entity=entity,
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

        self.callback = WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2,)

        self.kore_env = DummyVecEnv([self.make_env])

        self.model = A2C("MlpPolicy", self.kore_env, learning_rate=0.0008, verbose=1, tensorboard_log=f"runs/{run.id}")

    def make_env(self):
        return Monitor(self.unwrapped_env)

    def fit(self):
        replay_callback = ReplayCallback(self.step, interval=5, folder_name=self.name)
        self.model.learn(total_timesteps=2500000, callback=[self.callback, replay_callback])

    def step(self, obs, config):
        board = Board(obs, config)
        state = self.state_constr(board)
        agent_action = self.model.predict(state.tensor)[0]
        board_wrapper = BoardWrapper(board=board, player_id=0)

        return self.action_adapter.agent_to_kore_action(agent_action, board_wrapper)
