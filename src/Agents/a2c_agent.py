"""
This module implements the A2CAgent class
"""

import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv

from src.Agents.neural_networks.hybrid_cnn_mlp import HybridNet
from src.Agents.train_callbacks.game_stat_callback import GameStatCallback
from src.Agents.train_callbacks.replay_callback import ReplayCallback
from src.Monitoring.kore_monitor import KoreMonitor


class A2CAgent:
    """
    Wrapper of the a2c model from stable baselines for the kore task
    """

    def __init__(self, env: VecEnv, kore_monitor: KoreMonitor, n_training_steps: int = 1500000):
        self.name = "a2c"
        self.env = env
        self.monitor_callback = kore_monitor.callback
        self.n_training_steps = n_training_steps

        kore_monitor.set_run_name(self.name)

        policy_kwargs = {
            'features_extractor_class': HybridNet,
            'activation_fn': torch.nn.ReLU,
        }

        self.__model = A2C(
            policy="MultiInputPolicy",
            env=self.env,
            max_grad_norm=0.0005,
            learning_rate=0.0008,
            verbose=1,
            tensorboard_log=kore_monitor.tensorboard_log,
            policy_kwargs=policy_kwargs,
            gamma=1
        )

    def fit(self):
        """
        Wrapper of the models fit function
        """

        self.__model.learn(
            total_timesteps=self.n_training_steps,
            callback=[self.monitor_callback, GameStatCallback(), ReplayCallback(episodes_interval=50)],
        )
