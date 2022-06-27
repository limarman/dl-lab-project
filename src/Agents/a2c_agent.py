"""
This module implements the A2CAgent class
"""

import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv
from wandb.integration.sb3 import WandbCallback

from src.Agents.neural_networks.hybrid_cnn_mlp import HybridNet
from src.Agents.train_callbacks.game_stat_callback import GameStatCallback
from src.Agents.train_callbacks.replay_callback import ReplayCallback


class A2CAgent:
    """
    Wrapper of the a2c model from stable baselines for the kore task
    """

    def __init__(self, env: VecEnv, wandb_run, n_training_steps: int = 1500000):
        self.__env = env
        self.__n_training_steps = n_training_steps

        self.__wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{wandb_run.id}",
            verbose=2
        )

        self.__tensorboard_log = f"runs/{wandb_run.id}"

        policy_kwargs = {
            'features_extractor_class': HybridNet,
            'activation_fn': torch.nn.ReLU,
        }

        self.__model = A2C(
            policy="MultiInputPolicy",
            env=self.__env,
            max_grad_norm=0.0005,
            learning_rate=0.0008,
            verbose=1,
            tensorboard_log=self.__tensorboard_log,
            policy_kwargs=policy_kwargs,
        )

    def fit(self):
        """
        Wrapper of the models fit function
        """

        self.__model.learn(
            total_timesteps=self.__n_training_steps,
            callback=[self.__wandb_callback, GameStatCallback(), ReplayCallback(episodes_interval=10)],
        )

