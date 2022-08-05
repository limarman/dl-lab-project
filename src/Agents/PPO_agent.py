import os

import torch
# from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.vec_env import VecEnv

from src.Agents.neural_networks.hybrid_net import HybridNet, HybridResNet
from src.Agents.train_callbacks.game_stat_callback import GameStatCallback
from src.Agents.train_callbacks.replay_callback import ReplayCallback
from src.Monitoring.kore_monitor import KoreMonitor


class PPOAgent:
    """
    Wrapper of the PPO model from stable baselines for the kore task
    """

    def __init__(self, env: VecEnv,
                 kore_monitor: KoreMonitor,
                 run_id: str,
                 n_training_steps: int = 15,  #1500000
                 name='PPO',
                 resume_training=False,
                 feature_extractor_class: HybridNet=HybridResNet,
                 save_freq = 500,
                 selfplay=False):

        self.save_freq = save_freq
        self.selfplay = selfplay
        self.name = name
        self.env = env
        self.monitor_callback = kore_monitor.callback
        self.n_training_steps = n_training_steps
        self.run_id = run_id

        kore_monitor.set_run_name(self.name)

        policy_kwargs = {
            'features_extractor_class': feature_extractor_class,
            'activation_fn': torch.nn.ReLU,
        }

        if resume_training:
            self.model = PPO.load(f"checkpoints/{run_id}.zip", env=self.env)
        else:
            self.model = PPO(
                policy="MultiInputPolicy",
                env=self.env,
                learning_rate=0.0008,  # 0.0008
                verbose=1,
                tensorboard_log=kore_monitor.tensorboard_log,
                policy_kwargs=policy_kwargs,
                gamma=1
            )

        if self.selfplay:
            self.checkpoint_callback = CheckpointCallback(save_freq=self.save_freq, save_path=os.path.abspath(f"../selfplay_models/PPO/{self.run_id}")
                                                , name_prefix='selfplay', verbose=0)

        # periodically save model for later evaluation
        # self.checkpoint_callback = CheckpointCallback(save_freq=600,
        #                                               save_path=os.path.abspath(f"../backup_models/PPO/{self.run_id}")
        #                                               , name_prefix='Periodic_safe', verbose=0)

    def fit(self):
        """
        Wrapper of the models fit function
        """

        try:
            if self.selfplay:
                self.model.learn(
                    total_timesteps=self.n_training_steps,
                    callback=[self.monitor_callback, GameStatCallback(episodes_interval=10), ReplayCallback(episodes_interval=30), self.checkpoint_callback]  # 50
                )
            else:  # same but without checkpoint_callback
                self.model.learn(
                    total_timesteps=self.n_training_steps,
                    callback=[self.monitor_callback, GameStatCallback(episodes_interval=10), ReplayCallback(episodes_interval=30)]
                )

        finally:
            self.model.save(f"checkpoints/{self.run_id}")
            print("Saving checkpoint")
            print("DONE WITH LEARN")




