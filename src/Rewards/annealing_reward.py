from typing import Dict
from src.Rewards.win_reward import WinReward
from src.Rewards.advantage_reward import AdvantageReward
from src.States.kore_state import KoreState
from src.Rewards.kore_reward import KoreReward
import numpy as np
import src.core.global_vars


class AnnealingReward(KoreReward):
    """
    Starts from a dense reward and slowly anneals towards a sparse reward.
    """

    def __init__(self, first_reward=AdvantageReward(), last_reward=WinReward()):
        """
        First reward is an initial dense reward.
        Last reward is the sparse win reward
        """
        super().__init__()
        self.first_reward = first_reward
        self.last_reward = last_reward
        self.temp = 28000

        # initialize the step counter
        src.core.global_vars.init_global_step(0)  # add the step to start from, default = 0

    @staticmethod
    def get_reward_from_action(current_state: KoreState, actions):
        return None

    @staticmethod
    def get_reward_from_states(previous_state: KoreState, next_state: KoreState):
        return None

    def get_reward(self, previous_state: KoreState, next_state: KoreState, action: Dict[str, str]):
        """time is the number of time step we are currently at,
        temp is a constant hyperparameter,
        It also returns reward info now"""

        time = src.core.global_vars.return_step_count()

        first_reward_value, first_reward_info = self.first_reward.get_reward(previous_state, next_state, action)
        last_reward_value, last_reward_info = self.last_reward.get_reward(previous_state, next_state, action)
        E = np.e ** -(time / self.temp)
        annealing_reward = first_reward_value * E + last_reward_value * (1 - E)
        annealing_reward_info = {"E": E} | first_reward_info | last_reward_info

        return annealing_reward, annealing_reward_info

    def reset(self):
        pass
