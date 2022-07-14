"""
This module implements the A2CAgent class
"""
import os

import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv

from src.Agents.neural_networks.hybrid_net import HybridNet, HybridResNet
from src.Agents.train_callbacks.game_stat_callback import GameStatCallback
from src.Agents.train_callbacks.replay_callback import ReplayCallback
from src.Monitoring.kore_monitor import KoreMonitor


class A2CAgent:
    """
    Wrapper of the a2c model from stable baselines for the kore task
    """

    def __init__(self, env: VecEnv,
                 kore_monitor: KoreMonitor,
                 run_id: str,
                 n_training_steps: int = 1500000,
                 name='A2C', resume_training=False,
                 feature_extractor_class: HybridNet=HybridResNet):
        self.name = name
        self.env = env
        self.monitor_callback = kore_monitor.callback
        self.n_training_steps = n_training_steps
        self.run_id = run_id

        kore_monitor.set_run_name(self.name)

        policy_kwargs = {
            'features_extractor_class': feature_extractor_class,
            'activation_fn': torch.nn.ReLU,
            'optimizer_class': torch.optim.Adam,
            # optimizer_kwargs: dict()
        }

        if resume_training:
            self.__model = A2C.load(f"checkpoints/{run_id}.zip", env=self.env)
        else:
            self.__model = A2C(
                policy="MultiInputPolicy",
                env=self.env,
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
        try:
            self.__model.learn(
                total_timesteps=self.n_training_steps,
                callback=[self.monitor_callback, GameStatCallback(episodes_interval=10), ReplayCallback(episodes_interval=100)],
            )
        finally:
            self.__model.save(f"checkpoints/{self.run_id}")

