import math

from src.Rewards.KoreReward import KoreReward
from src.States.KoreState import KoreState
from src.States.SimpleState import SimpleState


class SimpleReward(KoreReward):
    """
    Simple reward implementation for quick prototyping
    """

    def __init__(self):
        """
        TODO add some weights/params here
        """
        pass

    @staticmethod
    def get_reward_from_action(current_state: SimpleState, actions) -> float:
        """
        Executes the actions and calculates a reward based on the state changes

        :param current_state: current state
        :param action: single action
        :return: scalar reward
        """
        next_state = current_state.apply_action_to_board(action)

        # TODO define more advanced reward function
        # TODO move it to superclass ?
        return SimpleReward.get_reward_from_states(current_state, next_state)

    @staticmethod
    def get_reward_from_states(previous_state: SimpleState, next_state: SimpleState):
        kore_delta = previous_state.kore_me - next_state.kore_me
        ship_delta = previous_state.ship_count_me - next_state.ship_count_me

        # increasing ships is more valuable in the beginning than the end
        weight_decay_ships = math.exp(1 - previous_state.step_normalized) * (1 / math.exp(1))
        reward = kore_delta + weight_decay_ships * ship_delta * 10

        return reward
