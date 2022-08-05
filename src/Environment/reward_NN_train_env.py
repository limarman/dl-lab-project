import uuid

from .kore_env import KoreEnv
from gym.core import ActType, ObsType
from typing import Tuple, Union, Callable
import torch
import os
from src.Actions.action_adapter import ActionAdapter
from src.Rewards.kore_reward import KoreReward
import shutil
import src.core.global_vars


class RewardNNTrainEnv(KoreEnv):
    """Wrapper over the KoreEnv class with the additional ability to save the labeled states.
    This is later used to train a neural network to predict the win probability of a game."""

    def __init__(self, state_constr, action_adapter: ActionAdapter, kore_reward: KoreReward, enemy_agent):
        super().__init__(state_constr, action_adapter, kore_reward, enemy_agent)

        self.state_save_frequency = 10  # frequency of episode with which to safe the state (all states of one episode are saved)

        self.temp_folder = os.path.abspath("../state_tensors/states/temp")
        self.win_folder = os.path.abspath("../state_tensors/states/win")
        self.lose_folder = os.path.abspath("../state_tensors/states/lose")

        # initialize the global variables
        src.core.global_vars.init_global_episode()

        # empty the temp folder (in case program was stopped prematurely and there are some files in the temp folder)
        for f in list(os.listdir(self.temp_folder)):
            os.remove(os.path.join(self.temp_folder, f))

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        next_state_tensor, reward, done, info = super(RewardNNTrainEnv, self).step(action)

        # save state for each step

        useful_tensor = {'maps': next_state_tensor['maps'][[0, 1, 4, 5, 12, 13]],  # we only select some maps from the state (look at board_wrapper)
                         'scalars': next_state_tensor['scalars']}

        if src.core.global_vars.return_episode_number() % self.state_save_frequency == 0 and src.core.global_vars.return_episode_number() > 50:
            torch.save(useful_tensor, os.path.join(self.temp_folder, f'state_{str(uuid.uuid1())}.pt'))

        # if done, check win or not, place the tensors in the appropriate folders
        if done:

            # increment the global episode count
            src.core.global_vars.increment_episode_number()

            # check if win or lose
            kore_delta = info['kore_delta']
            my_fleets = info['fleet_count_me']
            my_shipyards = info['shipyard_count_me']
            opponent_shipyards = info['shipyard_count_opponent']
            opponent_fleets = info['fleet_count_opponent']

            win = 1 if ((kore_delta > 0 and (my_fleets > 0 or my_shipyards > 0))
                        or (opponent_shipyards == 0 and opponent_fleets == 0)) else 0

            # moving files based on win
            if win:
                target_folder = self.win_folder
            else:
                target_folder = self.lose_folder

            state_files = list(os.listdir(self.temp_folder))
            for state_file in state_files:
                shutil.move(os.path.join(self.temp_folder, state_file), target_folder)

        return next_state_tensor, reward, done, info

