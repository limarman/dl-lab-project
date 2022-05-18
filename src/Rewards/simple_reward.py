import math

from src.Rewards.kore_reward import KoreReward
from src.States.simple_state import SimpleState


class SimpleReward(KoreReward):
    """
    Simple reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """

    @staticmethod
    def get_reward_from_action(state: SimpleState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = state.apply_action_to_board(actions)

        # TODO define more advanced reward function
        return SimpleReward.get_reward_from_states(state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: SimpleState, next_state: SimpleState):
        kore_delta = previous_state.kore_me - next_state.kore_me
        ship_delta = previous_state.ship_count_me - next_state.ship_count_me

        # increasing ships is more valuable in the beginning than the end
        weight_decay_ships = math.exp(1 - previous_state.step_normalized) * (1 / math.exp(1))
        reward = kore_delta + weight_decay_ships * ship_delta * 10

        return reward
