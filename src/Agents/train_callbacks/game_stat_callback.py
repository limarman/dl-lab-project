from collections import defaultdict, Counter

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
        self.prefix = 'custom'

    def _init_callback(self) -> None:
        self.actions = defaultdict(list)
        self.none_actions = defaultdict(list)
        wandb.define_metric('episodes')
        wandb.define_metric(self.prefix + '/*', step_metric='episodes')
        self.episodes = 0
        self.step = 0

    def _on_step(self) -> bool:
        """
        End of game statistics and the action frequencies are logged to wandb

        :return: If the callback returns False, training is aborted early.
        """
        game_stats = {}

        # We get from locals arrays with length = number of processes used
        for idx, (done, infos) in enumerate(zip(self.locals['dones'], self.locals['infos'])):
            self.actions[idx] += infos['actions']
            self.none_actions[idx] += infos['none_actions']

            if done:
                self.episodes += 1
                game_stats['Game Length'] = infos['game_length']
                game_stats['Kore Delta (at End)'] = infos['kore_delta']
                game_stats['Shipyard Count me (at End)'] = infos['shipyard_count_me']
                game_stats['Shipyard Count Opponent (at End)'] = infos['shipyard_count_opponent']
                game_stats['Kore me (at End)'] = infos['kore_me']
                game_stats['Fleet Count me (at End)'] = infos['fleet_count_me']
                game_stats['Fleet Count Opponent (at End)'] = infos['fleet_count_opponent']
                kore_delta = infos['kore_delta']
                my_fleets = infos['fleet_count_me']
                my_shipyards = infos['shipyard_count_me']
                win = 1 if (kore_delta > 0 and (my_fleets > 0 or my_shipyards > 0)) else 0
                game_stats['Winrate'] = win

                counts = Counter(self.actions[idx])
                action_infos = {'Frequency of ' + key: counts[key] / len(self.actions[idx]) for key in counts}
                action_infos['Frequency of None Actions'] = sum(self.none_actions[idx]) / len(self.none_actions[idx])
                game_stats.update(action_infos)
                self.actions[idx] = []
                self.none_actions[idx] = []

                # add prefix (needed to log on episodes)
                log_dict = {self.prefix + '/' + k: v for k, v in game_stats.items()}
                log_dict['episodes'] = self.episodes

                wandb.log(log_dict)

        return True
