from typing import Dict

from src.Rewards.kore_reward import KoreReward
from src.States.kore_state import KoreState


class WinReward(KoreReward):
    """
    Dummy reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """

    @staticmethod
    def get_reward_from_action(current_state: KoreState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(actions)
        return WinReward.get_reward_from_states(current_state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: KoreState, next_state: KoreState):
        kore_delta = next_state.kore_me - previous_state.kore_me

        return max(kore_delta, 0)

    def get_reward(self, previous_state: KoreState, next_state: KoreState, action: Dict[str, str]):
        return next_state.board_wrapper.game_result()
