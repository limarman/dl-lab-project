from typing import Dict
from src.Rewards.win_reward import WinReward
from src.Rewards.advantage_reward import AdvantageReward
from src.States.kore_state import KoreState
from src.Rewards.kore_reward import KoreReward

import torch
from src.Agents.neural_networks.hybrid_net_win_loss import HybridNet
from src.Agents.neural_networks.cnn_pytorch import BasicCNN

import os
import numpy as np
import src.core.global_vars


class NNReward(KoreReward):
    """
    Use a pretrained NN to get the reward. The NN outputs 1 if it thinks the agent is going to win and 0 otherwise.
    """

    def __init__(self, model_path=os.path.abspath("../state_tensors/model_7.pth")):
        """
        First reward is an initial dense reward.
        Last reward is the sparse win reward
        """
        self.softmax = torch.nn.Softmax(dim=1)
        self.dim_dict = {'maps': 6,
                         'scalars': 24}

        self.reward_nn = HybridNet(self.dim_dict, BasicCNN)
        self.model_path = model_path

        self.reward_nn.load_state_dict(torch.load(self.model_path))
        self.reward_nn.eval()

        super().__init__()

    @staticmethod
    def get_reward_from_action(current_state: KoreState, actions):
        return None

    @staticmethod
    def get_reward_from_states(previous_state: KoreState, next_state: KoreState):
        return None

    def get_reward(self, previous_state: KoreState, next_state: KoreState, action: Dict[str, str]):
        """Modify the state to get the "useful state". Useful state only contains some boards. Check boardwrapper class.
        Input the modified state to the rewardNN. Use the output of the NN as (part of) reward function.
        The reward is only non-zero if the model thinks we are more likely to win."""

        next_tensor = next_state.tensor

        useful_tensor = {'maps': next_tensor['maps'][[0, 1, 4, 5, 12, 13]],
                         # we only select some maps from the state (see board_wrapper class)
                         'scalars': next_tensor['scalars']}

        useful_maps = torch.tensor(useful_tensor['maps']).unsqueeze(0).float()
        useful_scalars = torch.tensor(useful_tensor['scalars']).unsqueeze(0).float()

        softmax = torch.nn.Softmax(dim=-1)

        output = softmax(self.reward_nn(useful_maps, useful_scalars))

        # my_win_prob = output[0][0].item()  # apply softmax, get the 1st (and only) batch, get the first player's win probability
        #
        # if my_win_prob > output[0][1].item():  # check if we are more likely to win than the opponent
        #     reward = my_win_prob
        # else:
        #     reward = 0

        reward = output[0][0].item()

        return reward, {}

    def reset(self):
        pass
