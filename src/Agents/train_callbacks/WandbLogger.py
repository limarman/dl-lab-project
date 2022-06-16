import os
import timeit
import warnings

import numpy as np
import wandb
from rl.callbacks import Callback


class WandbLogger(Callback):
    """ Similar to TrainEpisodeLogger, but sends data to Weights & Biases to be visualized """

    def __init__(self, name='dqn_agent'):
        project = "rl-dl-lab"
        entity = os.environ.get("WANDB_ENTITY")
        name = name

        wandb.init(project=project, name=name, entity=entity, allow_val_change=True)
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.kore_me = {}
        self.game_length = {}
        self.kore_delta = {}
        self.shipyards_me = {}
        self.shipyards_op = {}
        self.none_actions = {}
        self.step = 0

    def on_train_begin(self, logs):
        self.train_start = timeit.default_timer()
        self.metrics_names = self.model.metrics_names
        wandb.config.update({
            'params': self.params,
            'env': self.env.__dict__,
            'env.env': self.env.__env.__dict__,
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
        self.kore_me[episode] = []
        self.game_length[episode] = []
        self.kore_delta[episode] = []
        self.shipyards_me[episode] = []
        self.shipyards_op[episode] = []
        self.none_actions[episode] = []

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
            'Game Length': self.game_length[episode][-1],
            'My Kore (at end) ': self.kore_me[episode][-1],
            'Kore Delta (at end)': self.kore_delta[episode][-1],
            'My Shipyard Count (at end)': self.shipyards_me[episode][-1],
            'Opponent Shipyard Count (at end)': self.shipyards_op[episode][-1],
            'Won': 1 if (self.kore_delta[episode][-1] > 0 and self.shipyards_me[episode][-1] > 0) else 0,
            'Actions':  wandb.Histogram(np_histogram=np.histogram(self.actions[episode], density=True)),
            'Invalid Action Frequency': sum(self.none_actions[episode])/len(self.actions[episode]),
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
        del self.kore_me[episode]
        del self.game_length[episode]
        del self.shipyards_me[episode]
        del self.shipyards_op[episode]
        del self.none_actions[episode]
        del self.kore_delta[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.kore_me[episode].append(logs['info']['kore_me'])
        self.game_length[episode].append(logs['info']['game_length'])
        self.kore_delta[episode].append(logs['info']['kore_delta'])
        self.shipyards_me[episode].append(logs['info']['shipyard_count_me'])
        self.shipyards_op[episode].append(logs['info']['shipyard_count_opponent'])
        self.none_actions[episode].append(logs['info']['none_actions'])
        self.step += 1

