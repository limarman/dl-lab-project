import os
import timeit
import warnings

import numpy as np
import wandb
from rl.callbacks import Callback


class WandbLogger(Callback):
    """ Similar to TrainEpisodeLogger, but sends data to Weights & Biases to be visualized """

    def __init__(self):
        project = "rl-dl-lab"
        entity = os.environ.get("WANDB_ENTITY")
        name = "dqn_agent"

        wandb.init(project=project, name=name, entity=entity)
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        wandb.config.update({
            'params': self.params,
            'env': self.env.__dict__,
            'env.env': self.env.env.__dict__,
            #'env.env.spec': self.env.env.spec.__dict__,
            'agent': self.model.__dict__
        })

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and log training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        metrics = np.array(self.metrics[episode])
        metrics_dict = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    metrics_dict[name] = np.nanmean(metrics[:, idx])
                except Warning:
                    metrics_dict[name] = float('nan')

        wandb.log({
            'step': self.step,
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            #'obs_mean': np.mean(self.observations[episode]),
            #'obs_min': np.min(self.observations[episode]),
            #'obs_max': np.max(self.observations[episode]),
            **metrics_dict
        })

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1
