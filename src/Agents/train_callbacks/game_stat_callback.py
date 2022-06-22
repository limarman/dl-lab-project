from collections import defaultdict, Counter
from typing import Dict, Any

import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GameStatCallback(BaseCallback):
    """
    Logs action frequencies and end of game statisics to wandb

    :param verbose:
    """

    def __init__(self, verbose: int = 0, action_interval=10):
        """
        :param action_interval specifies the length of the step interval
        before pushing aggregated action frequencies to wandb
        """
        super().__init__()
        self.action_interval = action_interval

    def _init_callback(self) -> None:
        self.actions = []
        self.none_actions = []

    def _on_step(self) -> bool:
        """
        End of game statistics and the action frequencies are logged to wandb

        :return: If the callback returns False, training is aborted early.
        """
        num_done = 0
        game_stats = defaultdict(list)

        # We get from locals arrays with length = number of processes used
        for done, infos in zip(self.locals['dones'], self.locals['infos']):
            if done:
                num_done += 1
                game_stats['Game Length'].append(infos['game_length'])
                game_stats['Kore Delta (at End)'].append(infos['kore_delta'])
                game_stats['Shipyard Count me (at End)'].append(infos['shipyard_count_me'])
                game_stats['Shipyard Count Opponent (at End)'].append(infos['shipyard_count_opponent'])
                game_stats['Kore me (at End)'].append(infos['kore_me'])
                game_stats['Fleet Count me (at End)'].append(infos['fleet_count_me'])
                game_stats['Fleet Count Opponent (at End)'].append(infos['fleet_count_opponent'])
                kore_delta = infos['kore_delta']
                my_fleets = infos['fleet_count_me']
                my_shipyards = infos['shipyard_count_me']
                win = 1 if (kore_delta > 0 and (my_fleets > 0 or my_shipyards > 0)) else 0
                game_stats['Winrate'].append(win)

        # only report something if at least the game of one process is finished
        if num_done > 0:
            wandb.log({k: np.mean(v) for k, v in game_stats.items()})

        # always collect action infos
        for infos in self.locals['infos']:
            self.actions += infos['actions']
            self.none_actions += infos['none_actions']

        # push the action frequencies based on steps and not game ends
        # to guarantee proportional results
        if self.n_calls % self.action_interval == 0:
            counts = Counter(self.actions)
            action_infos = {'Frequency of ' + key: counts[key]/len(self.actions) for key in counts}
            action_infos['Frequency of None Actions'] = sum(self.none_actions)/len(self.none_actions)
            wandb.log(action_infos)
            self.actions = []
            self.none_actions = []

        return True


