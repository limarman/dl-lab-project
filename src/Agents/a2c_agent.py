"""
This module implements the A2CAgent class
"""

import os
import wandb
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv
from wandb.integration.sb3 import WandbCallback


class A2CAgent:
    """
    Wrapper of the a2c model from stable baselines for the kore task
    """

    def __init__(self, env: VecEnv, n_training_steps: int = 1500000):
        self.__env = env
        self.__n_training_steps = n_training_steps

        entity = os.environ.get("WANDB_ENTITY")
        run = wandb.init(
            project="rl-dl-lab",
            entity=entity,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

        self.__wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2
        )

        self.__tensorboard_log = f"runs/{run.id}"
        self.__model = A2C(
            policy="MlpPolicy",
            env=self.__env,
            learning_rate=0.0008,
            verbose=1,
            tensorboard_log=self.__tensorboard_log,
        )

    def fit(self):
        """
        Wrapper of the models fit function
        """

        self.__model.learn(
            total_timesteps=self.__n_training_steps,
            callback=[self.__wandb_callback],
        )
